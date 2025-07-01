import streamlit as st
import pandas as pd
import unicodedata
import io
from datetime import datetime
from cryptography.fernet import Fernet
import importlib.util
import sys
import re
import openpyxl
import os

# Decrypt and load BPA_models
def load_encrypted_module():
    try:
        key = os.environ.get('BPA_MODEL_KEY')
        if not key:
            st.error("环境变量 BPA_MODEL_KEY 未设置，请在 Streamlit Cloud 的 Secrets 设置中配置密钥")
            raise ValueError("环境变量 BPA_MODEL_KEY 未设置")
        cipher = Fernet(key.encode('utf-8'))
        with open('BPA_models.encrypted', 'rb') as f:
            encrypted = f.read()
        code = cipher.decrypt(encrypted).decode('utf-8')
        spec = importlib.util.spec_from_loader('BPA_models', loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules['BPA_models'] = module
        exec(code, module.__dict__)
        return module
    except Exception as e:
        st.error(f"无法解密 BPA_models.encrypted: {e}")
        raise

# Load BPA_models and BCard
try:
    BPA_models = load_encrypted_module()
    BCard = BPA_models.BCard
except Exception as e:
    st.error(f"加载 BPA_models 失败: {e}")
    raise

# PFO Parsing Functions
class PowerFlowRecord:
    def __init__(self, bus_name: str, rated_voltage: str, actual_voltage: str, dist: str, owner: str, unallocated_q: str = None):
        self.bus_name = bus_name
        self.rated_voltage = rated_voltage
        self.actual_voltage = actual_voltage
        self.dist = dist
        self.owner = owner
        self.unallocated_q = unallocated_q

    def to_dict(self) -> dict:
        return {
            'BusName': self.bus_name,
            'RatedVoltage': self.rated_voltage,
            'ActualVoltage': self.actual_voltage,
            'Dist': self.dist,
            'Owner': self.owner,
            'UnallocatedReactivePower': self.unallocated_q
        }

def read_pfo_file(file_content: bytes) -> list:
    try:
        for encoding in ['gbk', 'utf-8', 'latin1']:
            try:
                content = file_content.decode(encoding, errors='ignore')
                lines = content.splitlines()
                if lines:
                    return lines
            except UnicodeDecodeError:
                continue
        st.error("无法解码文件：尝试了 GBK、UTF-8 和 Latin1 编码均失败")
        return []
    except Exception as e:
        st.error(f"无法读取文件: {e}")
        return []

def find_bus_sections(lines: list) -> list:
    bus_endings = ['B', 'BQ', 'BE', 'BD', 'BA', 'BS', 'BM', '-PQ']
    sections = []
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Check if line ends with a bus ending or contains it (allowing for trailing data)
        if any(ending in line_stripped[-10:] for ending in bus_endings):
            sections.append(i)
    return sections

def extract_actual_voltage(line: str) -> str:
    if 'kV/' in line:
        kv_index = line.index('kV/')
        return line[kv_index - 7:kv_index].strip()
    return ''

def parse_pfo_data(lines: list) -> list:
    records = []
    bus_sections = find_bus_sections(lines)
    if not bus_sections:
        return records

    for i in bus_sections:
        line = lines[i]
        line_bytes = line.encode('gbk', errors='ignore')
        bus_name = line_bytes[0:8].decode('gbk', errors='ignore').strip()
        rated_voltage = line_bytes[8:14].decode('gbk', errors='ignore').strip()
        dist = line_bytes[36:38].decode('gbk', errors='ignore').strip()
        owner = line_bytes[38:40].decode('gbk', errors='ignore').strip()
        actual_voltage = extract_actual_voltage(line)
        unallocated_q = None

        # Check the current and subsequent lines for unallocated reactive power
        for j in range(i, min(i + 10, len(lines))):  # Limit scan to avoid excessive looping
            current_line = lines[j].strip()
            if '未安排无功' in current_line:
                try:
                    # Extract the number before "未安排无功" using regex to handle negative numbers and decimals
                    match = re.search(r'([-]?\d+\.\d+|\d+\.\d+)\s*未安排无功', current_line)
                    if match:
                        unallocated_q = float(match.group(1))
                        st.session_state.logs.append(f"[DEBUG] Found unallocated reactive power for {bus_name}: {unallocated_q}")
                    else:
                        st.session_state.logs.append(f"[DEBUG] Failed to parse unallocated reactive power in line: {current_line}")
                    break
                except (ValueError, IndexError) as e:
                    st.session_state.logs.append(f"[DEBUG] Error parsing unallocated reactive power for {bus_name}: {e}")
                    break
            # Stop if we hit another bus section
            if any(current_line.endswith(ending) or ending in current_line[-10:] for ending in ['B', 'BQ', 'BE', 'BD', 'BA', 'BS', 'BM', '-PQ']):
                if j != i:  # Don't break on the current bus line
                    break

        try:
            rated_voltage_float = float(rated_voltage)
            actual_voltage_float = float(actual_voltage) if actual_voltage else None
        except (ValueError, TypeError):
            st.session_state.logs.append(f"[DEBUG] Invalid voltage for {bus_name}: rated={rated_voltage}, actual={actual_voltage}")
            continue

        if actual_voltage_float is not None:
            record = PowerFlowRecord(bus_name, rated_voltage, actual_voltage, dist, owner, unallocated_q)
            records.append(record)
            st.session_state.logs.append(f"[DEBUG] Added record for {bus_name}, unallocated_q={unallocated_q}")

    return records

def check_voltage_anomalies(records: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_nodes_df = pd.DataFrame([record.to_dict() for record in records])
    if all_nodes_df.empty:
        return all_nodes_df, all_nodes_df

    all_nodes_df['RatedVoltage'] = pd.to_numeric(all_nodes_df['RatedVoltage'], errors='coerce')
    all_nodes_df['ActualVoltage'] = pd.to_numeric(all_nodes_df['ActualVoltage'], errors='coerce')
    all_nodes_df['UnallocatedReactivePower'] = pd.to_numeric(all_nodes_df['UnallocatedReactivePower'], errors='coerce')
    all_nodes_df = all_nodes_df.dropna(subset=['RatedVoltage', 'ActualVoltage'])

    def classify_voltage(row):
        rated = row['RatedVoltage']
        actual = row['ActualVoltage']
        if abs(rated - 500.0) < 30:  # 500 kV nodes (including 525 kV)
            min_v = 500.0
            max_v = 500.0 * 1.10
            alert_min_v = 500.0 * 1.052
            if actual < min_v:
                status = 'Low'
                deviation = (min_v - actual) / min_v * 100
            elif actual > max_v:
                status = 'High'
                deviation = (actual - max_v) / max_v * 100
            elif actual >= alert_min_v - 0.1 and actual <= max_v + 0.1:
                status = 'Alert High'
                deviation = (actual - alert_min_v) / alert_min_v * 100
            else:
                status = 'Normal'
                deviation = 0.0
            return status, deviation
        elif abs(rated - 230.0) < 25.0:  # 220 kV nodes
            min_v = 220.0 * 0.95
            max_v = 220.0 * 1.10
            if actual < min_v:
                status = 'Low'
                deviation = (min_v - actual) / min_v * 100
            elif actual > max_v:
                status = 'High'
                deviation = (actual - max_v) / max_v * 100
            else:
                status = 'Normal'
                deviation = 0.0
            return status, deviation
        return 'Excluded', 0.0

    all_nodes_df[['Status', 'Deviation (%)']] = all_nodes_df.apply(classify_voltage, axis=1, result_type='expand')
    all_nodes_df['Deviation (%)'] = all_nodes_df['Deviation (%)'].round(2)
    anomalies_df = all_nodes_df[all_nodes_df['Status'].isin(['Low', 'High', 'Alert High'])].copy()
    return all_nodes_df, anomalies_df

def _format_string(value, length):
    def char_width(char):
        ea_width = unicodedata.east_asian_width(char)
        return 2 if ea_width in ('F', 'W', 'A') else 1

    def string_width(string):
        return sum(char_width(c) for c in string)

    formatted_value = ''
    current_width = 0
    for char in str(value):
        if string_width(formatted_value) >= length:
            break
        w = char_width(char)
        if current_width + w > length:
            break
        formatted_value += char
        current_width += w

    while string_width(formatted_value) < length:
        formatted_value += ' '
    return formatted_value

class DATModifierApp:
    def __init__(self):
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'voltage_anomalies' not in st.session_state:
            st.session_state.voltage_anomalies = None
        if 'all_nodes' not in st.session_state:
            st.session_state.all_nodes = None
        self.logs = st.session_state.logs
        self.uploaded_files = st.session_state.uploaded_files
        self.b_parameters = {
            "shunt_var": "num",
        }

    def log(self, msg, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {msg}"
        self.logs.append(log_message)
        st.session_state.logs = self.logs
        try:
            with open("operation_log.txt", "a", encoding='utf-8') as log_file:
                log_file.write(log_message + "\n")
        except Exception as e:
            print(f"无法写入日志文件: {e}")

    def log_file_upload(self, file):
        if file is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_size = len(file.getvalue()) / 1024
            log_message = f"[{timestamp}] [UPLOAD] File uploaded: {file.name}, Size: {file_size:.2f} KB"
            self.logs.append(log_message)
            self.uploaded_files.append({
                "name": file.name,
                "size_kb": file_size,
                "timestamp": timestamp
            })
            st.session_state.logs = self.logs
            st.session_state.uploaded_files = self.uploaded_files
            try:
                with open("operation_log.txt", "a", encoding='utf-8') as log_file:
                    log_file.write(log_message + "\n")
            except Exception as e:
                print(f"无法写入日志文件: {e}")

    def read_and_parse_dat(self, file_content):
        self.log("读取文件内容")
        try:
            lines = file_content.decode('gbk', errors='ignore').splitlines()
        except Exception as e:
            st.error(f"无法读取文件: {e}")
            self.log(f"错误: 无法读取文件: {e}", level="ERROR")
            return None, None

        original_lines = []
        categorized_objects = {'B': []}

        for idx, line in enumerate(lines):
            line_stripped = line.rstrip('\n')
            if line_stripped.startswith("B "):
                try:
                    b_obj = BCard(line_stripped, idx)
                    categorized_objects['B'].append(b_obj)
                    original_lines.append(b_obj)
                except Exception as e:
                    self.log(f"错误: 解析 B卡失败 (行 {idx}): {e}", level="ERROR")
            else:
                original_lines.append(line_stripped)

        self.log("文件解析完成。")
        return original_lines, categorized_objects

    def write_back_dat(self, original_lines):
        output = io.StringIO()
        for item in original_lines:
            if hasattr(item, 'gen') and callable(item.gen):
                output.write(item.gen() + '\n')
            else:
                output.write(item + '\n')
        return output.getvalue().encode('gbk')

    def modify_b_cards(self, categorized_objects, dist_f, owner_f, vol_f, modifications):
        b_list = categorized_objects.get('B', [])
        filtered = []

        user_vol = None
        if vol_f:
            try:
                user_vol = float(vol_f)
            except ValueError:
                self.log(f"警告: B卡电压 '{vol_f}' 非法，忽略电压筛选", level="WARNING")
                user_vol = None

        dist_list = None
        if dist_f:
            dist_list = [item.strip() for item in re.split(r',|，', dist_f) if item.strip()]
            if not dist_list:
                self.log(f"警告: B卡分区 '{dist_f}' 格式非法，无有效值", level="WARNING")

        owner_list = None
        if owner_f:
            owner_list = [item.strip() for item in re.split(r',|，', owner_f) if item.strip()]
            if not owner_list:
                self.log(f"警告: B卡所有者 '{owner_f}' 格式非法，无有效值", level="WARNING")

        for b_obj in b_list:
            if dist_list and b_obj.dist.strip() not in dist_list:
                continue
            if owner_list and b_obj.owner.strip() not in owner_list:
                continue
            if user_vol is not None:
                b_vol = getattr(b_obj, "vol_rank", "0")
                try:
                    b_vol_f = float(b_vol)
                    if abs(b_vol_f - user_vol) >= 0.1:
                        continue
                except ValueError:
                    continue
            filtered.append(b_obj)

        self.log(f"B卡符合条件: {len(filtered)}")

        for b_obj in filtered:
            bus_name = getattr(b_obj, 'bus_name', 'NoBusName?')
            for param, mod in modifications.items():
                if not mod['apply']:
                    continue
                old_val = getattr(b_obj, param, "0")
                try:
                    old_val_num = float(old_val)
                    if mod['method'] == "set":
                        new_num = float(mod['value'])
                        setattr(b_obj, param, f"{new_num:.2f}")
                        self.log(f"B卡 [BusName={bus_name}]: {param} 设为 {new_num}")
                    else:
                        coeff = float(mod['value'])
                        new_num = old_val_num * coeff
                        setattr(b_obj, param, f"{new_num:.2f}")
                        self.log(f"B卡 [BusName={bus_name}]: {param} 由 {old_val_num}×{coeff}={new_num}")
                except ValueError:
                    self.log(f"错误: 无法将 B卡 [BusName={bus_name}] 的 {param}='{old_val}' 转为浮点数", level="ERROR")

    def create_b_shunt_var_tab(self):
        st.markdown("""
        **使用说明**:
        - 上传 PSD-BPA 格式的 `.dat` 文件以修改 B 卡的并联无功 (shunt_var)。
        - 在“筛选条件”中输入分区、所有者和电压等级（可选），用英文逗号 (,) 或中文逗号 (，) 分隔多个值。
        - 在“修改字段”中设置 shunt_var 的新值或乘系数。
        - 点击“执行修改”生成修改后的文件，点击“下载”保存。
        """)
        st.subheader("文件选择 (B卡 - shunt_var)")
        b_input_file = st.file_uploader("上传输入.dat文件", type=["dat"], key="b_input")
        if b_input_file:
            self.log_file_upload(b_input_file)
        b_output_filename = st.text_input("输出.dat文件名", value="modified_b_shunt_var.dat", key="b_output_filename")

        st.subheader("筛选条件")
        col1, col2, col3 = st.columns(3)
        with col1:
            b_dist = st.text_input("分区(dist, 用逗号分隔, 如 C1,D1)", value="C1,D1", key="b_dist")
        with col2:
            b_owner = st.text_input("所有者(owner, 用逗号分隔, 如 苏,锡)", value="苏,锡", key="b_owner")
        with col3:
            b_vol = st.text_input("电压(vol_rank)", value="", key="b_vol_rank")

        st.subheader("修改 shunt_var")
        modifications = {}
        param = "shunt_var"
        with st.expander("修改 shunt_var"):
            apply = st.checkbox("启用 shunt_var 修改", key=f"b_{param}_apply")
            if apply:
                method = st.radio(
                    "shunt_var 修改方式",
                    ["设值", "乘系数"],
                    key=f"b_{param}_method",
                    format_func=lambda x: x
                )
                method = "set" if method == "设值" else "mul"
                if method == "set":
                    value = st.text_input("shunt_var 新值", key=f"b_{param}_value")
                else:
                    value = st.text_input("shunt_var 系数", key=f"b_{param}_coeff")
                modifications[param] = {'apply': True, 'method': method, 'value': value}
            else:
                modifications[param] = {'apply': False, 'method': None, 'value': None}

        if st.button("执行修改", key="b_execute", type="primary"):
            if not b_input_file:
                st.warning("请选择输入的 .dat 文件。")
                return
            if not b_output_filename:
                st.warning("请指定输出文件名。")
                return

            self.log("开始处理 B卡 shunt_var...")
            original_lines, categorized = self.read_and_parse_dat(b_input_file.read())
            if original_lines is None or categorized is None:
                return
            self.modify_b_cards(categorized, b_dist, b_owner, b_vol, modifications)
            output_data = self.write_back_dat(original_lines)
            st.download_button(
                label="下载修改后的文件",
                data=output_data,
                file_name=b_output_filename,
                mime="application/octet-stream",
                key="b_download"
            )
            self.log(f"修改完成，准备下载: {b_output_filename}")

    def create_voltage_monitoring_tab(self):
        st.markdown("""
        **使用说明**:
        - 上传 PSD-BPA 格式的 `.pfo` 文件以监测节点电压异常和未安排无功。
        - 电压规范：
          - 500 kV 节点（标称电压可能为 525 kV）：
            - 正常范围：500–550 kV
            - 预警高压：526–550 kV（需关注但不计为异常）
            - 低于 500 kV 为异常低压
            - 高于 550 kV 为异常高压
          - 220 kV 节点（标称 230 kV）：正常范围 209–242 kV
          - 低于 220 kV 的节点不监测
        - 未安排无功：检查哪些节点存在未安排无功（MVar），并列出其值。
        - 查看异常、预警节点和未安排无功列表及分区/所有者分布，下载异常报告或完整节点数据为 Excel 文件。
        """)
        try:
            import openpyxl
        except ImportError:
            st.error("缺少 openpyxl 库，无法生成 Excel 文件。请安装：`pip install openpyxl`")
            return

        st.subheader("文件选择 (电压监测)")
        pfo_input_file = st.file_uploader("上传输入.pfo文件", type=["pfo"], key="pfo_input")
        if pfo_input_file:
            self.log_file_upload(pfo_input_file)
            st.session_state.voltage_anomalies = None
            st.session_state.all_nodes = None
        output_filename_anomalies = st.text_input("异常报告输出文件名", value="voltage_anomalies.xlsx", key="pfo_output_filename_anomalies")
        output_filename_all = st.text_input("完整节点数据输出文件名", value="all_nodes.xlsx", key="pfo_output_filename_all")

        if st.button("执行电压监测", key="pfo_execute", type="primary"):
            if not pfo_input_file:
                st.warning("请选择输入的 .pfo 文件。")
                return
            if not output_filename_anomalies or not output_filename_all:
                st.warning("请指定所有输出文件名。")
                return

            self.log("开始处理 PFO 文件进行电压监测...")
            lines = read_pfo_file(pfo_input_file.read())
            if not lines:
                self.log("错误: 无法解析 PFO 文件", level="ERROR")
                return

            records = parse_pfo_data(lines)
            if not records:
                st.warning("未找到有效的母线数据。")
                self.log("警告: 未找到有效的母线数据", level="WARNING")
                return

            all_nodes_df, anomalies_df = check_voltage_anomalies(records)
            if all_nodes_df.empty:
                st.success("未检测到任何节点数据。")
                self.log("电压监测完成：未检测到节点数据")
                st.session_state.voltage_anomalies = None
                st.session_state.all_nodes = None
                return

            st.session_state.voltage_anomalies = anomalies_df
            st.session_state.all_nodes = all_nodes_df

        # Display Results if Available
        if st.session_state.voltage_anomalies is not None or st.session_state.all_nodes is not None:
            anomalies_df = st.session_state.voltage_anomalies
            all_nodes_df = st.session_state.all_nodes

            # Unallocated Reactive Power
            st.subheader("未安排无功节点")
            unallocated_df = all_nodes_df[all_nodes_df['UnallocatedReactivePower'].notnull()]
            if not unallocated_df.empty:
                st.write(f"检测到 **{len(unallocated_df)}** 个节点存在未安排无功")
                st.dataframe(unallocated_df[['BusName', 'RatedVoltage', 'ActualVoltage', 'UnallocatedReactivePower', 'Dist', 'Owner']],
                             use_container_width=True)
            else:
                st.info("未检测到存在未安排无功的节点")

            # 500 kV Anomalies and Alerts
            st.subheader("500 kV 节点电压状态")
            if anomalies_df is not None:
                df_500kv = anomalies_df[abs(anomalies_df['RatedVoltage'] - 500.0) < 30]
                df_500kv_abnormal = df_500kv[df_500kv['Status'].isin(['Low', 'High'])]
                if not df_500kv_abnormal.empty:
                    st.write(f"检测到 **{len(df_500kv_abnormal)}** 个 500 kV 节点异常（低压或高压）")
                    st.dataframe(df_500kv_abnormal[['BusName', 'RatedVoltage', 'ActualVoltage', 'Status', 'Deviation (%)', 'Dist', 'Owner', 'UnallocatedReactivePower']],
                                 use_container_width=True)
                else:
                    st.info("未检测到 500 kV 节点电压异常")

                df_500kv_alert = df_500kv[df_500kv['Status'] == 'Alert High']
                if not df_500kv_alert.empty:
                    st.write(f"检测到 **{len(df_500kv_alert)}** 个 500 kV 节点预警高压（526–550 kV）")
                    st.dataframe(df_500kv_alert[['BusName', 'RatedVoltage', 'ActualVoltage', 'Status', 'Deviation (%)', 'Dist', 'Owner', 'UnallocatedReactivePower']],
                                 use_container_width=True)
                else:
                    st.info("未检测到 500 kV 节点预警高压")

                # 220 kV Anomalies
                st.subheader("220 kV 节点电压异常")
                df_220kv = anomalies_df[abs(anomalies_df['RatedVoltage'] - 230.0) < 25.0]
                if not df_220kv.empty:
                    st.write(f"检测到 **{len(df_220kv)}** 个 220 kV 节点异常")
                    st.dataframe(df_220kv[['BusName', 'RatedVoltage', 'ActualVoltage', 'Status', 'Deviation (%)', 'Dist', 'Owner', 'UnallocatedReactivePower']],
                                 use_container_width=True)
                else:
                    st.info("未检测到 220 kV 节点电压异常")

                # Summary by Dist and Owner
                st.subheader("异常及预警分布")
                col1, col2 = st.columns(2)
                with col1:
                    dist_summary = anomalies_df.groupby('Dist').size().reset_index(name='Count')
                    st.write("按分区 (Dist) 分布")
                    st.dataframe(dist_summary, use_container_width=True)
                with col2:
                    owner_summary = anomalies_df.groupby('Owner').size().reset_index(name='Count')
                    st.write("按所有者 (Owner) 分布")
                    st.dataframe(owner_summary, use_container_width=True)

                # Download Anomalies Report
                output_buffer = io.BytesIO()
                anomalies_df.to_excel(output_buffer, index=False)
                output_buffer.seek(0)
                if not output_filename_anomalies.endswith('.xlsx'):
                    output_filename_anomalies += '.xlsx'
                st.download_button(
                    label="下载异常报告",
                    data=output_buffer,
                    file_name=output_filename_anomalies,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="pfo_download_anomalies"
                )
                self.log(f"电压监测完成，异常报告准备下载: {output_filename_anomalies}")

            # Download All Nodes Report
            if all_nodes_df is not None:
                output_buffer_all = io.BytesIO()
                all_nodes_df.to_excel(output_buffer_all, index=False)
                output_buffer_all.seek(0)
                if not output_filename_all.endswith('.xlsx'):
                    output_filename_all += '.xlsx'
                st.download_button(
                    label="下载完整节点数据",
                    data=output_buffer_all,
                    file_name=output_filename_all,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="pfo_download_all"
                )
                self.log(f"电压监测完成，完整节点数据准备下载: {output_filename_all}")

    def main(self):
        st.set_page_config(page_title="BPA潮流无功自动优化系统", layout="wide")
        st.title("BPA潮流无功自动优化系统")
        st.markdown("""
            **注**: 本应用不会保留您上传的任何数据。
            结合BPA潮流计算的无功收敛过程，以下是使用本软件的推荐步骤：
            1. **确认有功收敛**：
               - 发现BPA数据在微调后，发生不收敛，首先确保BPA潮流计算已实现有功功率收敛，倒退至最近一次收敛状态，生成.pfo文件。
               - **注**：.dat文件的设置语句需要将P_OUTPUT_LIST设置为FULL，即/P_OUTPUT_LIST,FULL\，以使pfo文件输出完整的潮流计算结果。
            2. **分析电压异常和未安排无功**：
               - 如果你的.dat数据相应的.dxt文件的地理接线信息是完整的，可以使用BPA地理接线图模块中的电压等高线热力图直观监测电压异常的节点。否则，可进行如下步骤：
               - 在“电压监测”标签上传.pfo文件，执行电压监测。
               - 查看500 kV和220 kV节点的异常（Low/High）和500 kV警戒高压（Alert High）表格，以及未安排无功的节点。
               - 记录异常节点和未安排无功节点的Dist、Owner和BusName，以及建议调整方向：
                 - 低压（Low）：通常需要增加shunt_var（正值，表示更多电容）。
                 - 高压（High）或警戒高压（Alert High）：可能需要减少shunt_var（负值，表示更多电感）。
                 - 未安排无功：可能需要调整shunt_var以分配这些无功，方向取决于电压状态。
            3. **调整shunt_var**：
               - 切换到“B卡并联无功修改”标签，上传原始.dat文件。
               - 输入异常节点或未安排无功节点的dist、owner和vol_rank（一般修改37kV的无功装接节点）进行筛选。
               - 设置shunt_var修改：
                 - 对于低压节点或高未安排无功，尝试“设值”（如100）或“乘系数”（如1.2）增加容性无功。
                 - 对于高压节点，尝试“设值”（如-50）或“乘系数”（如0.8或者-1.0）减少容性无功或调整为感性无功。
               - 下载修改后的.dat文件。
            4. **验证收敛**：
               - 使用修改后的.dat文件运行BPA潮流计算。
               - 生成新的.pfo文件，重复电压监测，检查异常和未安排无功是否减少。
               - 必要时迭代调整shunt_var值，直至无功收敛。
            5. **分析分布**：
               - 查看“异常及预警分布”中的Dist和Owner统计，确定问题集中的区域，优化后续调整。
        """)

        tabs = st.tabs(["B卡并联无功修改", "电压监测"])
        with tabs[0]:
            self.create_b_shunt_var_tab()
        with tabs[1]:
            self.create_voltage_monitoring_tab()

        with st.expander("查看日志"):
            st.markdown("**日志说明**: 显示所有操作记录，包括文件上传、修改和监测结果。")
            st.text_area("操作日志 (可滚动查看，不可编辑)", value="\n".join(st.session_state.logs), height=200, key="log_output_main")

if __name__ == "__main__":
    app = DATModifierApp()
    app.main()
