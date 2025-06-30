import streamlit as st
import pandas as pd
import numpy as np

# 页面标题
st.title("风电出力率分析工具")

# 季节和时段定义
st.markdown("""
**季节定义**：
- 春季：3月、4月、5月
- 夏季：6月、7月、8月
- 秋季：9月、10月、11月
- 冬季：12月、1月、2月

**时段**：
- 19:00-22:00
- 11:00-14:00
- 1:00-4:00
""")

# 文件上传
st.subheader("1. 上传 CSV 文件")
uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])

# 初始化数据
columns = []
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='gbk')
        columns = df.columns.tolist()
        st.success("文件上传成功！")
        st.write("检测到的列名：", columns)
    except UnicodeDecodeError:
        st.error("文件编码错误，请尝试 UTF-8 或 GB2312 编码的 CSV 文件")
    except Exception as e:
        st.error(f"读取文件失败: {str(e)}")

# 参数配置
if columns:
    st.subheader("2. 配置分析参数")
    
    # 列选择
    st.markdown("**选择数据列**")
    col1, col2, col3 = st.columns(3)
    with col1:
        time_col = st.selectbox("时间列", columns, help="选择时间戳列（格式如 2023-03-15 19:00:00）")
    with col2:
        value_col = st.selectbox("值列（发电量）", columns, help="选择发电量列")
    with col3:
        unit_col = st.selectbox("机组编号列", columns, help="选择机组编号列")

    # 值列单位和总容量
    st.markdown("**单位与容量**")
    col1, col2 = st.columns(2)
    with col1:
        value_unit = st.selectbox("值列单位", ["kWh", "MWh"], help="发电量数据的单位")
    with col2:
        total_capacity = st.number_input(
            f"总容量（{value_unit}）",
            min_value=0.0,
            value=51.0 if value_unit == "MWh" else 51000.0,
            step=0.1,
            help="输入所有机组的总容量"
        )

    # 采样周期和分位数
    st.markdown("**采样周期与分位数**")
    col1, col2 = st.columns(2)
    with col1:
        sample_period_options = {
            "1小时": 60,
            "30分钟": 30,
            "15分钟": 15
        }
        sample_period_label = st.selectbox("采样周期", list(sample_period_options.keys()), help="选择数据的时间间隔")
        sample_period = sample_period_options[sample_period_label]
    with col2:
        percentile = st.number_input(
            "分位数（%）",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="例如，5 表示前5%分位出力率"
        )

    # 验证采样周期
    if df is not None and time_col in df.columns and unit_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            # 按机组和时间排序，计算每个机组的时间差
            df_sorted = df.sort_values([unit_col, time_col])
            time_diff = df_sorted.groupby(unit_col)[time_col].diff().dropna().dt.total_seconds() / 60
            common_period = time_diff.mode()[0] if not time_diff.empty else None
            if common_period and abs(common_period - sample_period) > 1e-6:
                st.warning(
                    f"检测到数据采样周期约为 {int(common_period)} 分钟，"
                    f"与选择的 {sample_period_label}（{sample_period} 分钟）不匹配，请确认！"
                )
            else:
                st.info(f"采样周期验证通过：约为 {sample_period} 分钟")
        except Exception as e:
            st.error(f"时间列格式错误: {str(e)}")

# 计算逻辑
def get_season(month):
    if month in [3, 4, 5]:
        return '春季'
    elif month in [6, 7, 8]:
        return '夏季'
    elif month in [9, 10, 11]:
        return '秋季'
    elif month in [12, 1, 2]:
        return '冬季'
    return None

def is_in_time_slot(time, time_slot, sample_period):
    hour = time.hour
    if sample_period == 60:
        if time_slot == '19:00-22:00':
            return 19 <= hour <= 22
        elif time_slot == '11:00-14:00':
            return 11 <= hour <= 14
        elif time_slot == '1:00-4:00':
            return 1 <= hour <= 4
    else:
        minute = time.minute
        if time_slot == '19:00-22:00':
            return (19 <= hour < 22) or (hour == 22 and minute == 0)
        elif time_slot == '11:00-14:00':
            return (11 <= hour < 14) or (hour == 14 and minute == 0)
        elif time_slot == '1:00-4:00':
            return (1 <= hour < 4) or (hour == 4 and minute == 0)
    return False

def calculate_power_rate(df, time_col, value_col, unit_col, value_unit, total_capacity, sample_period, percentile):
    time_slots = ['19:00-22:00', '11:00-14:00', '1:00-4:00']
    seasons = ['春季', '夏季', '秋季', '冬季']
    results = []
    
    # 单位转换
    if value_unit == "MWh":
        total_capacity_kW = total_capacity * 1000
        value_to_kW = lambda x: x * 1000
    else:
        total_capacity_kW = total_capacity
        value_to_kW = lambda x: x
    
    # 采样周期（小时）
    period_in_hours = sample_period / 60.0
    
    for season in seasons:
        df_season = df[df[time_col].dt.month.apply(lambda x: get_season(x) == season)]
        
        for slot in time_slots:
            df_slot = df_season[df_season[time_col].apply(lambda x: is_in_time_slot(x, slot, sample_period))]
            
            if not df_slot.empty:
                # 聚合多机组数据
                total_power = df_slot.groupby(time_col)[value_col].sum()
                
                if sample_period != 60:
                    total_power = total_power.resample(f'{sample_period}T').sum()
                
                # 转换为功率（kW）
                total_power = value_to_kW(total_power) / period_in_hours
                power_rate = total_power / total_capacity_kW
                sorted_power_rate = power_rate.sort_values(ascending=False)
                top_percentile = np.percentile(sorted_power_rate, 100 - percentile) if not sorted_power_rate.empty else np.nan
            else:
                top_percentile = np.nan
            
            results.append({
                '季节': season,
                '时段': slot,
                f'{percentile}%概率出力率': top_percentile
            })
    
    return pd.DataFrame(results)

# 运行分析
if st.button("运行分析", type="primary"):
    if df is not None and time_col and value_col and unit_col and total_capacity > 0:
        with st.spinner("正在分析数据..."):
            try:
                results_df = calculate_power_rate(
                    df, time_col, value_col, unit_col, value_unit, total_capacity, sample_period, percentile
                )
                st.subheader("3. 分析结果")
                st.dataframe(
                    results_df.style.format({f'{percentile}%概率出力率': '{:.4f}'}, na_rep='无有效数据'),
                    use_container_width=True
                )
                # 下载结果
                csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="下载结果为 CSV",
                    data=csv,
                    file_name="power_rate_results.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"分析失败: {str(e)}")
    else:
        st.error("请确保已上传文件、选择所有列、输入有效容量！")

# 调试信息
if st.checkbox("显示调试信息", help="查看数据预览和分布，用于排查问题"):
    if df is not None:
        st.subheader("调试信息")
        st.write("**数据预览**：")
        st.dataframe(df.head(20))
        st.write("**时间列小时分布**：")
        st.write(df[time_col].dt.hour.value_counts().sort_index())
        st.write("**季节分布**：")
        st.write(df[time_col].dt.month.apply(get_season).value_counts())
        st.write("**机组数量**：")
        st.write(df.groupby(time_col)[unit_col].nunique().value_counts())rcentile}%概率出力率': '{:.4f}'}, na_rep='无有效数据'))
        except Exception as e:
            st.error(f"分析过程中发生错误: {str(e)}")
    else:
        st.error("请确保已上传文件、选择所有列、输入有效容量！")