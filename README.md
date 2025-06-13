# 哆啦A梦与WALL-E图像分类应用

这是一个使用 FastAI 训练的图像分类应用，可以识别哆啦A梦和WALL-E图像。

## 功能特点

- 支持上传 JPG、JPEG、PNG 格式的图片
- 使用 FastAI 训练的深度学习模型进行图像分类
- 显示预测结果和置信度

## 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
streamlit run 03-streamlit-walle-doraemon.py
```

## 部署到 Streamlit Cloud

1. 将代码推送到 GitHub 仓库
2. 在 [Streamlit Cloud](https://streamlit.io/cloud) 创建新应用
3. 连接到你的 GitHub 仓库
4. 指定主文件：`03-streamlit-walle-doraemon.py`

## 注意事项

- 确保模型文件 `doraemon_walle_model.pkl` 存在于项目根目录
- 模型文件较大，建议使用 Git LFS 管理 