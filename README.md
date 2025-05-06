大二春季计算机视觉课的期末作业。关于可解释性模型的功能验证和对比。
解释器：layer-cam,grad_cam,occlusion,random_masking
模型：ResNet50,MobileNetV2
数据集：cifar-10
环境见requirements.txt
运行顺序：先运行main.py,再运行analyze_results进行结果分析。
其实还贴了其他解释器：如score_cam,lime explainer,shap explainer和其他模型，但是由于数据集和算力的限制，主函数和结果分析只调用了上文的解释器和模型。有机会可以改一下代码进行改进，讲其他模型也验证一下。cifar-10数据集对score_cam的效果很差，所以放弃了使用。
