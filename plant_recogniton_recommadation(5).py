import streamlit as st
import sys
import pandas as pd


# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

from fastai.vision.all import *
import pathlib

# 模型加载函数
@st.cache_resource
def load_model():
    """加载并缓存模型"""
    model_metadata = {  # 添加元数据字典
    'model_type': 'FastAI CNN',
    'input_size': (3, 224, 224),
    'batch_size': 64,
    'framework': 'FastAI'
    }
    temp = None
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
    
    try:
        # 图像分类模型
        image_model = load_learner(pathlib.Path(__file__).parent / "植物病害识别.pkl")
        # 协同过滤模型（新增调试日志）
        collab_path = pathlib.Path(__file__).parent / "植物推荐系统.pkl"
        # 验证文件存在性
        if not collab_path.exists():
            raise FileNotFoundError(f"协同过滤模型文件不存在于：{collab_path}")
        # 验证文件大小（调整为更合理的阈值）
        file_size = collab_path.stat().st_size
        if file_size < 10240:  # 调整为10KB阈值
            raise ValueError(f"模型文件过小，可能不完整")
        with open(collab_path, 'rb') as f:
            # 添加weights_only参数解决警告
            collab_data = torch.load(
                f, 
                map_location='cpu',
                weights_only=False  # 显式声明加载模式
            )
        return {
            'image': image_model,
            'collab': {
                'user_emb': collab_data['user_emb'],
                'item_emb': collab_data['item_emb'],
                'plant_ids': collab_data['plant_ids']
            },
            'metadata': model_metadata  # 添加元数据字段
        }
    except Exception as e:
        st.error(f"""模型加载失败: {str(e)}""")
        return None
    finally:
        if sys.platform == "win32" and temp is not None:
            pathlib.PosixPath = temp

@st.cache_data
def load_plant_intros():
    """加载植物介绍数据"""
    try:
        # 明确指定列名并添加格式验证
        df = pd.read_excel('plant_intros.xlsx', 
                         header=0,
                         names=['plant', 'intro'],  # 强制指定列名
                         usecols=[0, 1])  # 只读取前两列
        
        # 添加列存在性检查
        if not {'plant', 'intro'}.issubset(df.columns):
            missing = {'plant', 'intro'} - set(df.columns)
            raise ValueError(f"缺少必要字段: {missing}")
            
        return df
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return pd.DataFrame()
        
@st.cache_data
def load_treatments():
    """加载防治方案数据"""
    try:
        return pd.read_excel('防治.xlsx').set_index('疾病')
    except Exception as e:
        st.error(f"防治方案加载失败: {str(e)}")
        return pd.DataFrame()

# 推荐功能
@st.cache_data
def load_plant_features():
    """加载植物特征数据"""
    try:
        return pd.read_csv('u.item', sep='|', encoding='utf-8', 
                         names=[
                             'plant_id', 'plant','好养护','价格实惠','款式多','生长速度快','耐旱','耐寒','耐阴','耐水',
                             '耐光耐晒','易开花','容易徒长泛滥','适合新手','易繁殖','风水价值','观赏价值','毒性成分',
                             '寓意美好','改善空气','缓解焦虑改善睡眠','用途广泛','抗病虫害','土壤要求严格','空间需求大',
                             '阳光要求高','水分要求高','花叶异味'
                         ])
        plants['plant'] = plants['plant'].str.strip()# 名称清洗
        intros = load_plant_intros()
        if not intros.empty:
            intros = intros.reset_index()  
            plants = plants.merge(
                intros,
                on='plant',
                how='left',
                validate='m:1'
            )
        plants['intro'] = plants.get('intro', '暂无详细介绍')
        plants['intro'] = plants['intro'].fillna('暂无详细介绍')
            
        return plants
    except Exception as e:
        st.error(f"特征数据加载失败: {str(e)}")
        return pd.DataFrame()



@st.cache_data
def load_popular_plants():
    """加载并计算热门植物排行榜"""
    try:
        # 加载特征数据（新增去重逻辑）
        df = pd.read_csv('u.item', sep='|', encoding='utf-8',
                       names=[
                           'plant_id', 'plant','好养护','价格实惠','款式多','生长速度快','耐旱','耐寒','耐阴','耐水',
                           '耐光耐晒','易开花','容易徒长泛滥','适合新手','易繁殖','风水价值','观赏价值','毒性成分',
                           '寓意美好','改善空气','缓解焦虑改善睡眠','用途广泛','抗病虫害','土壤要求严格','空间需求大',
                           '阳光要求高','水分要求高','花叶异味'
                       ]).drop_duplicates('plant_id')  # 新增去重
        
        # 加载评分数据
        ratings = pd.read_csv('u.data', sep='\t', 
                           names=['user_id','plant_id','rating','timestamp'])
        
        # 合并数据（新增验证参数）
        df = df.merge(
            ratings.groupby('plant_id')['rating'].mean().round(1).reset_index(),
            on='plant_id',
            how='left',
            validate='1:1'  # 新增验证
        ).fillna({'rating': 3.0})  # 无评分默认3分
            
        # 计算综合得分（特征分+评分）
        positive_features = [
            '好养护', '价格实惠', '款式多', '生长速度快', '耐旱',
            '耐寒', '耐阴', '耐水', '耐光耐晒', '易开花',
            '适合新手', '易繁殖', '改善空气', '抗病虫害'
        ]
        
        df['feature_score'] = df[positive_features].sum(axis=1)
        df['综合得分'] = df['feature_score'] * 0.7 + df['rating'] * 0.3 * 2
        
        # 按综合得分排序
        return df.sort_values('综合得分', ascending=False).drop_duplicates('plant').head(5)
        
    except Exception as e:
        st.sidebar.error(f"数据加载失败: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_ratings():
    """加载用户评分数据"""
    try:
        return pd.read_csv('u.data', sep='\t', 
                        names=['user_id','plant_id','rating','timestamp'])
    except Exception as e:
        st.error(f"评分数据加载失败: {str(e)}")
        return pd.DataFrame()

def init_session():
    """初始化session state"""
    if 'plant_init' not in st.session_state:
        st.session_state.plant_init = True
        st.session_state.current_step = 1
        st.session_state.selected_plants = []
        st.session_state.user_ratings = {}

def cold_start_recommendation(user_features, plants_df, top_n=5):
    """基于用户选择特征的冷启动推荐"""
    try:
        # 合并评分数据
        ratings = load_ratings().groupby('plant_id')['rating'].mean().reset_index()
        plants_df = plants_df.merge(ratings, on='plant_id', how='left').fillna({'rating': 3.0})

        # 新增特征得分计算
        positive_features = [
            '好养护', '价格实惠', '款式多', '生长速度快', '耐旱',
            '耐寒', '耐阴', '耐水', '耐光耐晒', '易开花',
            '适合新手', '易繁殖', '改善空气', '抗病虫害'
        ]
        plants_df['feature_score'] = plants_df[positive_features].sum(axis=1)

        # 计算特征权重
        feature_weights = {feat: 1 for feat in user_features if feat}
        
        # 计算植物匹配度得分
        plants_df['cold_score'] = plants_df[feature_weights.keys()].mul(
            pd.Series(feature_weights)
        ).sum(axis=1)
        
        # 排除零分植物
        filtered = plants_df[plants_df['cold_score'] > 0]
        
       # 按评分和特征得分排序
        result = filtered.sort_values(
            ['rating', 'cold_score'], 
            ascending=False
        ).head(top_n)
        
        # 添加预测评分字段（使用历史评分的80%作为预测值）
        result['predicted_rating'] = result['rating']  # 移除0.8系数
        return result.sample(n=min(top_n, len(result)))

    except Exception as e:
        st.error(f"冷启动推荐失败: {str(e)}")
        return pd.DataFrame()

def hybrid_recommendation(user_features, user_ratings=None, top_n=80):
    """修改后的混合推荐函数"""
    try:
        plants = load_plant_features()
        ratings = load_ratings()
        
        # 修复特征处理逻辑
        if user_ratings:
            # 获取用户评分过的植物名称列表
            rated_plants = list(user_ratings.keys())
            # 从植物数据中提取这些植物的特征
            rated_features = plants[plants['plant'].isin(rated_plants)]
            # 检查是否存在已评分植物
            if not rated_features.empty:
                # 获取评分植物共有的积极特征（出现次数超过50%的特征）
                common_features = rated_features[[
                    '好养护','适合新手','易繁殖','改善空气','抗病虫害'
                ]].mean().gt(0.5).index.tolist()
                user_features = common_features

        
        # 生成用户ID（示例使用特征哈希）
        user_id = abs(hash(frozenset(user_features))) % (10**6)
        
        # 冷启动处理（新用户）
        if user_id not in ratings['user_id'].unique():
            return cold_start_recommendation(user_features, plants, top_n)
            
        # 创建数据加载器
        dls = CollabDataLoaders.from_df(
            ratings,
            user_name='user_id',
            item_name='plant_id',
            rating_name='rating',
            valid_pct=0.2,
            bs=64
        )
        
        # 在模型训练前添加随机因子配置
        learn = collab_learner(
            dls,
            n_factors=50,
            y_range=(0, 5.5),  # 调整预测范围上限
            metrics=[rmse, mae],
            model_dir="/tmp",
            wd=0.1
        )
        
        # 模型训练（使用1cycle策略）
        learn.fit_one_cycle(5, 5e-3, wd=0.1)
        
        # 生成预测
        all_plants = plants['plant_id'].unique()
        user_plants = pd.DataFrame({
            'user_id': [user_id] * len(all_plants),
            'plant_id': all_plants
        })
        
        # 获取预测评分
        dl = learn.dls.test_dl(user_plants)
        preds, _ = learn.get_preds(dl=dl)
        user_plants['predicted_rating'] = preds.numpy().flatten()
        
        # 修改结果排序逻辑
        result_df = user_plants.merge(
            plants[['plant_id', 'plant', 'feature_score', 'rating']],
            on='plant_id'
        )
        result_df['feature_percent'] = (result_df['feature_score'] / 14 * 100).round(1)
        return result_df.sort_values('predicted_rating', ascending=False).head(top_n)

    except Exception as e:
        st.error(f"推荐系统错误: {str(e)}")
        return pd.DataFrame() 
        
    except Exception as e:
        st.error(f"推荐系统错误: {str(e)}")
        return pd.DataFrame()

# 满意度计算函数
def calculate_satisfaction(ratings):
    """计算推荐满意度（基于1-5评分）"""
    if not ratings:
        return 0.0
    
    # 转换评分到0-100%范围
    valid_ratings = [r for r in ratings.values() if r > 0]
    if not valid_ratings:
        return 50.0  # 中性评分
    
    avg_rating = sum(valid_ratings) / len(valid_ratings)
    return (avg_rating / 5) * 100
    


# 修改CSS样式中的背景部分
st.markdown("""
<style>
     /* 全屏布局 */
    .stApp {
        background: linear-gradient(135deg, #f0fff0 0%, #d0f0d0 100%);
        padding: 0 !important;  margin: 0;  height: 100vh;
    }   

    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: #e0ffe0 !important;
        border-right: 2px solid #c8e6c9;
        text-align: center !important;
    }

    /* 按钮样式 */
    .stButton>button {
        background: #a5d6a7 !important;
        color: #1b5e20 !important;
        border: 1px solid #81c784;
    }

    /* 生长时间轴样式 */
    .growth-timeline {
        position: relative;
        padding: 20px 0;
        border-left: 2px solid var(--primary);
        margin-left: 20px;
    }
    .growth-stage {
        position: relative;
        margin: 30px 0;
        padding-left: 30px;
    }
    .growth-stage::before {
        content: '';  position: absolute;  left: -10px;
        top: 15px;    width: 20px;         height: 20px;
        background: var(--secondary);      border-radius: 50%;
        border: 3px solid var(--accent);
    }
    /* CSS变量 */
    :root {
        --primary: #1B5E20;      --secondary: #A5D6A7;     --accent: #81C784;
        --water-color: #4DB6AC;  --temp-color: #FFB74D;    --light-color: #FFF176;
        --growth-color: #AED581; --disease-color: #EF9A9A; --growth-color: #7CB342; 
        --environment-color: #689F38; 
    }

    /* 统一卡片基础样式 */
    .care-card {
        border-radius: 8px;  padding: 15px;  margin-bottom: 15px;  transition: all 0.3s ease;
        border: 2px solid var(--secondary);  background: linear-gradient(145deg, #f8fff8 0%, #e8f5e9 100%);
    }
    .care-card:hover {
        transform: translateY(-3px);   box-shadow: 0 4px 15px rgba(var(--primary),0.1);
    }
    .care-card[data-theme="growth"] {
        border-color: var(--growth-color);
        background: linear-gradient(145deg, #f1f8e9 0%, #dcedc8 100%);
    }
    .care-card[data-theme="environment"] {
        border-color: var(--environment-color);
        background: linear-gradient(145deg, #e8f5e9 0%, #c8e6c9 100%);
    }

    /* 通过data-theme属性设置卡片颜色 */
    .care-card[data-theme="water"] {
        border-color: var(--water-color);
        background: linear-gradient(145deg, #e0f7fa 0%, #b2ebf2 100%);
    }
    .care-card[data-theme="temperature"] {
        border-color: var(--temp-color);
        background: linear-gradient(145deg, #fff3e0 0%, #ffe0b2 100%);
    }
    .care-card[data-theme="light"] {
        border-color: var(--light-color);
        background: linear-gradient(145deg, #fffde7 0%, #fff9c4 100%);
    }

    /* 统一标签样式 */
    .care-tag {
        display: inline-block;
        padding: 4px 10px;
        margin: 0 8px 8px 0;
        border-radius: 12px;
        background: rgba(var(--primary),0.1);
        font-size: 0.9em;
    }

    .disease-table th {
        background: var(--primary);
        color: white;
    }
    .disease-table tr:nth-child(even) {
        background: rgba(var(--secondary),0.1);
    }
        /* 新增卡片高度控制样式 */
    .plant-card {
        height: 320px !important;
        overflow: hidden;
    }
    .plant-card img {
        height: 200px !important;
        object-fit: cover;
        width: 100%;
    }
    .plant-info {
        height: 120px !important;
        overflow-y: auto;
        padding: 10px !important;
    }
    
</style>
""", unsafe_allow_html=True)
  

# --------------主应用------------
st.title("🎄植物养护及病害识别系统")
model = load_model()

treatment_df = load_treatments()

# ================= 功能分区 =================
tab1, tab2, tab3 = st.tabs(["🌿 病害诊断", "🌱 植物推荐", "📚 知识百科"])
with tab1:
    uploaded_file = st.file_uploader("🫀请您上传植物叶片图片，我们将尽力为您解决问题", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            image = PILImage.create(uploaded_file)
            st.image(image, caption="上传的图片", use_container_width=True)
            
        with col2:
            pred, pred_idx, probs = model['image'].predict(image)
            st.success(f"🌿识别结果: {pred} (置信度: {probs[pred_idx]:.2%})")
            
            if not treatment_df.empty and pred in treatment_df.index:
                treatment = treatment_df.loc[pred]
                st.subheader("💊 防治方案")
                st.markdown(f"**📝症状描述**: {treatment['症状']}")
                for i in range(1, 9):
                    if pd.notna(treatment[f'防治{i}']):
                        st.markdown(f"{i}. {treatment[f'防治{i}']}")
            else:
                st.warning("暂未收录该病害的防治方案")

#用户交互式系统推荐
with tab2:
    init_session()
    
    # 步骤1：初始化推荐评分
    if st.session_state.current_step == 1:
        st.header("🍀 步骤1/4: 请为随机植物评分")
        
        if not st.session_state.selected_plants:
            plants = load_plant_features()
            st.session_state.selected_plants = plants[['plant', '好养护', '适合新手']].sample(3).to_dict('records')
        
        cols = st.columns(3)
        for idx, plant_info in enumerate(st.session_state.selected_plants):
            with cols[idx]:
                plant_name = plant_info['plant']
                st.subheader(f"植物 {idx+1}")
                
                # 添加带固定尺寸的图片容器
                with st.container():
                    try:
                        img_path = pathlib.Path(__file__).parent / "plants" / f"{plant_name}.jpg"
                        st.image(
                            str(img_path), 
                            caption=plant_name, 
                            use_container_width=True,  # 保持容器宽度
                            output_format="JPEG",  # 统一输出格式
                            width=300  # 设置统一宽度（像素）
                        )
                    except Exception as e:
                        st.warning("很抱歉，该植物图片暂未收录")
                        # 添加占位图保持布局统一
                        st.image("https://placehold.co/300x200?text=图片待补充", 
                               use_container_width=True)
                
                st.markdown(f"**名称**: {plant_name}")
                st.markdown(f"**养护难度**: {'★' * plant_info['好养护']} ({plant_info['好养护']}/5)")
                st.markdown(f"**新手推荐**: {'✅ 适合' if plant_info['适合新手'] else '❌ 不适合'}")
                 
                rating = st.slider(
                    f"对 {plant_name} 的评分",
                    1, 5, 3,
                    key=f"init_rating_{plant_name}"
                )
                st.session_state.user_ratings[plant_name] = rating

        if st.button("提交初始化评分", type="primary"):
            # 生成初始推荐（基于评分）
            st.session_state.initial_recommend = hybrid_recommendation(
                user_features=st.session_state.user_ratings.keys(),  # 使用评分植物作为特征
                user_ratings=st.session_state.user_ratings
            )
            st.session_state.current_step = 2
            st.rerun()

    # 步骤2显示初始推荐
    elif st.session_state.current_step == 2:
        st.header("🌼 步骤2/4: 初始推荐结果")
        
        if not st.session_state.initial_recommend.empty:
            st.success("根据您的评分生成以下推荐：")
            cols = st.columns(3)
            for idx, (_, row) in enumerate(st.session_state.initial_recommend.head(9).iterrows()):
                with cols[idx % 3]:
                    with st.expander(f"💐 {row['plant']} (预测评分: {row['predicted_rating']:.1f}/5)"):
                        st.metric("养护难度", f"{row['好养护']}/5")
                        st.progress(row['predicted_rating']/5)
                        st.caption(f"特征匹配度: {row['feature_score']}/14")
            
            # 满意度调查
            satisfaction = st.radio("您是否满意当前推荐？", 
                                ["✅ 满意🙂‍↕，结束推荐", "❌ 不满意🙂‍↔️，精准匹配"],
                                index=0,
                                key="satisfaction")
            
            if st.button("确认选择"):
                if "不满意" in satisfaction:
                    st.session_state.current_step = 3  # 进入特征选择
                    st.rerun()
                else:
                    st.session_state.initial_recommend = pd.DataFrame()
                    st.session_state.result_df = pd.DataFrame()
                    st.session_state.current_step = 5   
                    st.rerun() 
        else:
                # 添加更详细的错误提示
                st.warning("""
                    暂时无法生成推荐，可能原因：
                    1. 没有找到匹配的植物特征
                    2. 评分数据尚未加载完成
                    3. 推荐模型初始化失败
                    """)  

    # 步骤3：特征选择
    elif st.session_state.current_step == 3:
        st.header("🪴 步骤3/4: 选择偏好特征")
        st.info("为了进行更精准的匹配，建议最好选择您喜欢的至少三种植物类型")
        
        with st.form("recommend_form"):
            features = ['','好养护','价格实惠','款式多','生长速度快','耐旱','耐寒','耐阴','耐水',
                      '耐光耐晒','易开花','适合新手','易繁殖','改善空气','抗病虫害']
            selected = []
            cols_row1 = st.columns(3)
            for i in range(3):  # 0-2
                with cols_row1[i]:
                    selected.append(st.selectbox(
                        f"特征 {i+1}", 
                        features,
                        index=0,
                        key=f"feature_{i}"
                    ))
            
            cols_row2 = st.columns(3)
            for i in range(3,6):  # 3-5
                with cols_row2[i-3]:
                    selected.append(st.selectbox(
                        f"特征 {i+1}", 
                        features,
                        index=0,
                        key=f"feature_{i}"
                    ))
            
            if st.form_submit_button("✨生成推荐"):
                valid_features = [f for f in selected if f]
                if valid_features:
                    st.session_state.valid_features = valid_features
                    st.session_state.current_step = 4
                    st.rerun()
                else:
                    # 将错误提示移动到步骤3的表单内
                    st.error("请至少选择一个特征")
                    # 阻止进入下一步
                    st.session_state.current_step = 3
                    st.rerun()

    # 步骤4：推荐结果
    elif st.session_state.current_step == 4:
        st.header("🌳 步骤4/4: 个性化推荐")
        
        if 'valid_features' in st.session_state:
            # 添加加载提示和异常处理
            with st.spinner("正在生成推荐..."):
                try:
                    st.session_state.result_df = hybrid_recommendation(
                        user_features=st.session_state.valid_features,  # 修正参数名
                        user_ratings=st.session_state.user_ratings  # 添加评分数据
                    )
                except Exception as e:
                    st.error(f"推荐生成失败: {str(e)}")
            
            if not st.session_state.result_df.empty:
                st.success(f"找到 {len(st.session_state.result_df)} 种符合要求的植物，为您推荐前6种：")
                
                # 初始化评分存储
                if 'plant_ratings' not in st.session_state:
                    st.session_state.plant_ratings = {}
                
                # 为每个推荐项创建评分表单
                with st.form("plant_rating_form"):
                    for idx, row in st.session_state.result_df.head(6).iterrows():
                        with st.expander(f"🌴 {row['plant']} | 预测评分: {row['predicted_rating']:.1f}/5", expanded=True):
                            st.metric("养护优势匹配度", f"{row['feature_score']}/14")
                            st.progress(row['predicted_rating']/5)
                            
                            # 添加评分组件
                            rating = st.slider(
                                "请为本项推荐评分",
                                1, 5, 3,
                                key=f"plant_rating_{row['plant']}"
                            )
                            st.session_state.plant_ratings[row['plant']] = rating
                    
                    # 表单提交按钮
                    if st.form_submit_button("✨ 提交所有评分"):
                        # 计算满意度
                        satisfaction = calculate_satisfaction(st.session_state.plant_ratings)
                        st.session_state.satisfaction_result = {
                            'score': satisfaction,
                            'ratings': st.session_state.plant_ratings.copy()
                        }
                        st.session_state.current_step = 5  # 进入满意度分析
                        st.rerun()
    # 步骤5：推荐完成
    elif st.session_state.current_step == 5:
        st.header("🎉 推荐完成")
        st.success("已根据您的偏好完成推荐，欢迎再次使用！")
        
        # 添加动画触发状态标记
        if 'animation_shown' not in st.session_state:
            st.session_state.animation_shown = False
        
        if st.button("重新开始"):
            # 清除所有相关状态
            keys_to_remove = ['satisfaction_result', 'plant_ratings', 'result_df', 
                            'initial_recommend', 'valid_features', 'animation_shown']
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_step = 1
            st.rerun()

        # 满意度分析面板
        if 'satisfaction_result' in st.session_state:
            st.markdown("---")
            st.subheader("📊 推荐满意度分析")
            
            satisfaction = st.session_state.satisfaction_result['score']
            ratings = st.session_state.satisfaction_result['ratings']
            
            cols = st.columns([1, 3])
            with cols[0]:
                st.metric("综合满意度", f"{satisfaction:.1f}%")
                # 添加一次性动画触发逻辑
                if not st.session_state.animation_shown:
                    if satisfaction >= 80:
                        st.success("🥳 推荐效果很好！非常符合您的需求")
                        st.balloons()
                    elif satisfaction >= 60:
                        st.info("😊 推荐效果良好，比较满意")
                        st.balloons()
                    elif satisfaction >= 40:
                        st.warning("😐 推荐效果一般，我们会改进")
                        st.snow()
                    else:
                        st.error("😞 抱歉，我们会重新调整推荐策略")
                        st.snow()
                    st.session_state.animation_shown = True  # 
                st.caption("计算方法：")
                st.caption("• 平均评分/5 × 100%")
                st.caption("• 3分(60%)为中性评分")
            
            with cols[1]:
                st.subheader("详细评分记录")
                for plant, rating in ratings.items():
                    # 获取植物详细信息
                    plant_data = st.session_state.result_df[st.session_state.result_df['plant'] == plant].iloc[0]
                    
                    # 使用列布局
                    col_left, col_right = st.columns([2, 3])
                    with col_left:
                        st.markdown(f"**{plant.strip()}**：{'⭐' * rating}{'☆' * (5 - rating)}")
                    with col_right:
                        st.markdown(f"""
                        <div style="display: flex; gap: 15px;">
                            <span>特征匹配度: {plant_data['feature_score']}/14</span>
                            <span>养护难度: {'★' * plant_data['好养护']}</span>
                        </div>
                        """, unsafe_allow_html=True)                
    
with tab3:
    with st.container() as knowledge_container:
        st.markdown('<div class="knowledge-section">', unsafe_allow_html=True)
        st.header("🌏 植物百科")
        tab1, tab2, tab3 = st.tabs(["养护知识", "病害图鉴","植物介绍"])
        
        with tab1:
            # 养护知识分类标题
            st.markdown('<div class="care-category-title">', unsafe_allow_html=True)
            st.markdown('<h2>🌵 基础养护知识</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 养护知识卡片布局
            st.markdown('<div class="care-grid">', unsafe_allow_html=True)
            
            # 养护常识卡片
            st.markdown('''
            <div class="care-card">
                <h3>🎓 养护常识</h3>
                <ul>
                    <li>浇水见干见湿</li>
                    <li>保持通风良好</li>
                    <li>定期检查病虫害</li>
                    <li>避免强光直射</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)
            
            # 浇水指南卡片
            st.markdown('''
            <div class="care-card" data-theme="water">
                <h3>💧 浇水指南</h3>
                <div class="care-tag">春秋季：3-5天/次</div>
                <div class="care-tag">夏季：早晚各一次</div>
                <div class="care-tag">冬季：7-10天/次</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # 温度适宜卡片
            st.markdown('''
            <div class="care-card" data-theme="temperature">
                <h3>🌡️ 温度适宜</h3>
                <div class="care-tag">春秋季：15-25°C</div>
                <div class="care-tag">夏季：20-30°C</div>
                <div class="care-tag">冬季：10-20°C</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # 光照要求卡片
            st.markdown('''
            <div class="care-card" data-theme="light">
                <h3>🌞 光照要求</h3>
                <div class="care-tag">春秋季：6-8小时</div>
                <div class="care-tag">夏季：10-12小时</div>
                <div class="care-tag">冬季：4-6小时</div>
            </div>
            ''', unsafe_allow_html=True)
            
            #生长周期
            st.markdown('''
            <div class="care-card" data-theme="growth">
                <h3>🔄 生长周期</h3>
                <div class="care-tag">春秋季：6-12个月</div>
                <div class="care-tag">夏季：4-8个月</div>
                <div class="care-tag">冬季：10-14个月</div>
            </div>
            ''', unsafe_allow_html=True)
            
           # 生长环境
            st.markdown('''
            <div class="care-card" data-theme="environment">
                <h3>🫒 生长环境</h3>
                <div class="care-tag">春秋季：湿润、通风良好</div>
                <div class="care-tag">夏季：湿润、通风良好</div>
                <div class="care-tag">冬季：干燥、通风良好</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 高级布局分隔线
            st.markdown('<div class="layout-divider"></div>', unsafe_allow_html=True)
            
            # 植物生长阶段分类标题
            st.markdown('<div class="care-category-title">', unsafe_allow_html=True)
            st.markdown('<h2>🎍 植物生长阶段指南</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 植物生长阶段卡片
            st.markdown('''
            <div class="growth-timeline">
                <div class="growth-stage">
                    <h4>🌱 发芽期</h4>
                    <p>保持土壤湿润，避免阳光直射<br>温度控制在20-25°C</p>
                </div>
                <div class="growth-stage">
                    <h4>🌿 幼苗期</h4>
                    <p>逐渐增加光照，保持通风<br>适量施肥，温度18-25°C</p>
                </div>
                <div class="growth-stage">
                    <h4>🌳 生长期</h4>
                    <p>充足光照，定期浇水施肥<br>注意病虫害防治，温度15-28°C</p>
                </div>
                <div class="growth-stage">
                    <h4>🌸 花期</h4>
                    <p>增加磷钾肥，保持湿度<br>避免温度波动，温度18-25°C</p>
                </div>
                <div class="growth-stage">
                    <h4>🍎 结果期</h4>
                    <p>增加钾肥，控制浇水量<br>保证充足光照，温度15-25°C</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # 高级布局分隔线
            st.markdown('<div class="layout-divider"></div>', unsafe_allow_html=True)
            # 季节养护分类标题
            st.markdown('<div class="care-category-title">', unsafe_allow_html=True)
            st.markdown('<h2>🍃 四季养护指南</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            # 季节按钮组
            cols = st.columns(4)
            seasons = {
                "spring": "🎋 春",
                "summer": "☀️ 夏", 
                "autumn": "🍁 秋",
                "winter": "❄️ 冬"
            }
            for idx, (season, emoji) in enumerate(seasons.items()):
                with cols[idx]:
                    # 移除type参数避免触发特效
                    if st.button(
                        emoji,
                        key=f"season_{season}",
                        help=f"点击查看{season}季养护要点"
                    ):
                        st.session_state.active_season = season
            # 显示对应季节的内容
            if 'active_season' in st.session_state:
                season_content = {
                    "spring": {
                        "title": "🎋 春季养护要点",
                        "items": [
                            "逐渐增加浇水量",
                            "定期施肥，促进生长",
                            "修剪枯枝，促进新芽萌发",
                            "注意预防病虫害"
                        ]
                    },
                    "summer": {
                        "title": "☀️ 夏季养护要点",
                        "items": [
                            "避免强光直射，适当遮荫",
                            "早晚浇水，保持土壤湿润",
                            "增加通风，降低温度",
                            "控制施肥量，避免烧根"
                        ]
                    },
                    "autumn": {
                        "title": "🍁 秋季养护要点",
                        "items": [
                            "减少浇水量，促进植物休眠",
                            "增加磷钾肥，增强抗寒能力",
                            "清除落叶，预防病虫害",
                            "适当修剪，保持株型"
                        ]
                    },
                    "winter": {
                        "title": "❄️ 冬季养护要点",
                        "items": [
                            "减少浇水频率，保持土壤微干",
                            "停止施肥，让植物休眠",
                            "保持温暖，避免冻伤",
                            "增加光照时间"
                        ]
                    }
                }
                with st.expander(season_content[st.session_state.active_season]['title'], expanded=True):
                    st.markdown(f"""
                    <div class="care-card {st.session_state.active_season}">
                        <ul>
                            {"".join([f'<li>{item}</li>' for item in season_content[st.session_state.active_season]['items']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
        with tab2:
            # 病害图鉴分类标题
            st.markdown('<div class="care-category-title">', unsafe_allow_html=True)
            st.markdown('<h2>🩺 常见病害</h2>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 美化病害图鉴表格
            st.markdown('<div class="disease-table-container">', unsafe_allow_html=True)
            st.markdown('''
            <table class="disease-table">
                <thead>
                    <tr>
                        <th>病害名称</th><th>典型症状</th>
                        <th>易发季节</th><th>防治方法</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>白粉病</td><td>叶面白色粉末状覆盖物</td>
                        <td>春秋季</td><td>保持通风，喷施杀菌剂</td>
                    </tr>
                    <tr>
                        <td>黑斑病</td><td>褐色圆形斑点，边缘黄色</td>
                        <td>夏季</td><td>清除病叶，喷施杀菌剂</td>
                    </tr>
                    <tr>
                        <td>锈病</td><td>叶片背面出现橙黄色锈斑</td>
                        <td>夏季</td><td>保持干燥，喷施杀菌剂</td>
                    </tr>
                    <tr>
                        <td>炭疽病</td>
                        <td>叶片出现褐色圆形或不规则病斑</td>
                        <td>高温高湿季节</td>
                        <td>清除病叶，喷施杀菌剂</td>
                    </tr>
                    <tr>
                        <td>根腐病</td><td>根部腐烂，植株萎蔫</td>
                        <td>全年</td><td>控制浇水，更换土壤</td>
                    </tr>
                </tbody>
            </table>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            # 病害识别小贴士
            st.markdown('''
            <div class="care-card">
                <h3>💡 病害识别小贴士</h3>
                <ul>
                    <li>仔细观察病斑形状、颜色和分布</li>
                    <li>检查叶片背面和茎部</li>
                    <li>记录发病时间和环境条件</li>
                    <li>及时拍照保存病状特征</li>
                    <li>使用本系统上传照片进行智能识别</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)


        with tab3:
            st.header("🌻 植物百科大全")
            intros_df = load_plant_intros()
            
            if not intros_df.empty:
                if 'intro' not in intros_df.columns:
                    st.error("植物介绍数据格式错误，缺少'intro'字段")
                else:
                    search_term = st.text_input("🔍 搜索植物名称", help="输入关键词查找植物")
                    
                    # 使用瀑布流布局容器
                    with st.container():
                        cols = st.columns(3, gap="medium")
                        current_col = 0
                        
                        for idx, row in intros_df.iterrows():
                            plant_name = row['plant']
                            is_match = search_term.lower() in plant_name.lower() if search_term else True
                            
                            if not is_match:
                                continue
                            
                            with cols[current_col]:
                                # 卡片容器
                                with st.container():
                                    st.markdown(f"""
                                    <div style='
                                        background: linear-gradient(145deg, #f5fff5 0%, #e8f5e9 100%);
                                        border-radius: 12px;
                                        padding: 16px;
                                        margin: 12px 0;
                                        box-shadow: 0 3px 12px rgba(0,100,0,0.1);
                                        transition: transform 0.2s ease;
                                    '>
                                    """, unsafe_allow_html=True)
                                    
                                    # 使用两列布局（图片左，文字右）
                                    img_col, info_col = st.columns([5, 7])
                                    
                                    with img_col:  # 图片列
                                        try:
                                            img_path = pathlib.Path(__file__).parent / "plants" / f"{plant_name}.jpg"
                                            st.image(str(img_path), use_container_width=True)
                                        except:
                                            st.image("https://placehold.co/400x300?text=图片待更新", 
                                                   use_container_width=True)
                                    
                                    with info_col:  # 信息列
                                        st.markdown(f"##### {plant_name}")
                                        st.markdown(f"**📝 植物介绍** ")
                                        st.caption(f"✨ {row['intro'][:50]}")
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                            
                            current_col = (current_col + 1) % 3
                            card_key = f"plant_card_{idx}_{plant_name.strip()}"
                            

                            st.markdown("</div>", unsafe_allow_html=True)

#侧边栏
with st.sidebar:
    # 添加控制按钮
    if st.button("📌系统信息", use_container_width=True):
        st.session_state.show_info_center = not st.session_state.get('show_info_center', True)

    # 根据状态显示内容
    if st.session_state.get('show_info_center', True):
        st.markdown("### 模型状态")
        if model is not None:
            try:
                # 获取图像分类模型信息
            # 获取图像分类模型信息
                image_model = model['image']
                st.success("✅ 模型加载成功")
                st.markdown("### 模型信息")
                st.write(f"**模型类型**: {model['metadata']['model_type']}")
                st.write(f"**框架类型**: {model['metadata']['framework']}") 
                st.write(f"**输入图片尺寸**: {model['metadata']['input_size'][1]}x{model['metadata']['input_size'][2]}")
                st.write(f"**训练批次大小**: {model['metadata']['batch_size']}")
            except Exception as e:
                st.error(f" 模型信息解析失败: {str(e)}")
        else:
            st.error("❌ 模型未加载")

        # 显示数据统计
        st.markdown("---")
        st.markdown("### 数据统计")
        
        try:
            # 加载植物特征数据
            plants = load_plant_features()
            treatment_count = len(load_treatments())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("植物种类数", len(plants))
                st.metric("病害防治方案数", treatment_count)
                
            with col2:
                st.metric("特征维度", len(plants.columns)-1)
                st.metric("平均养护难度", plants['好养护'].mean().round(1))
                
        except Exception as e:
            st.error(f"数据加载错误: {str(e)}")
            st.markdown("---")
    st.title("🏆 最好养活植物推荐榜 🏆")
    popular_plants = load_popular_plants()
    
    if not popular_plants.empty:
        for idx, (_, row) in enumerate(popular_plants.iterrows(), 1):
            st.markdown(f"""  
            **{idx}. {row['plant']}**  
            🦋 匹配特征数：`{row['feature_score']}/14`  
            🌟 用户评分：{row["rating"]}/5 ({int(row["rating"]*20)}%)  
            🤗 新手友好：{'✅' if row['适合新手'] == 1 else '❌'}  
            """)
    else:
        st.warning("暂未生成排行榜数据")

st.markdown("---")
with st.expander("📮 用户反馈中心", expanded=False):
    tab1, tab2 = st.tabs(["📝 功能反馈", "✨ 推荐设置"])
    
    with tab1:
        with st.form("feedback_form"):
            feedback_type = st.radio(
                "反馈类型", 
                ["功能建议", "错误报告", "数据纠错"],
                horizontal=True
            )
            feedback_content = st.text_area("详细内容")
            if st.form_submit_button("提交反馈"):
                st.success("感谢您的反馈！我们将尽快处理")
        
            st.markdown("**🆘 紧急帮助**")
            st.page_link("https://www.example.com", label="联系植物医生", icon="📞")
    
    with tab2:
        st.info("推荐偏好设置（开发中）")
        st.progress(0.8)
        # 在回调函数中显示提示
        def show_thankyou_message():
            st.toast("感谢您的期待，我们会继续努力的！", icon="🎊")
        st.checkbox("优先显示易养护植物", True, on_change=show_thankyou_message)
        st.checkbox("隐藏有毒植物品种", on_change=show_thankyou_message)
        st.checkbox("仅显示适合新手的植物", on_change=show_thankyou_message)

    
             
st.caption("🌱 智能识别植物病害，守护绿色生命")


