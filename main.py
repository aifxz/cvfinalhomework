import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import time
import pandas as pd
from tabulate import tabulate

# 瀵煎叆棰勫鐞嗗嚱鏁�
from utils.preprocess import preprocess_for_cam, preprocess_for_lime, preprocess_for_shap
from utils.helpers import create_output_directories, setup_logging, get_device_info
from utils.visualize import plot_comparison

# 瀵煎叆瑙ｉ噴鍣
from explainers.grad_cam import run_grad_cam
from explainers.score_cam import run_score_cam
from explainers.layer_cam import run_layer_cam
from explainers.lime_explainer import lime_explanation
from explainers.shap_explainer import shap_explanation

# 瀵煎叆妯″瀷鍔犺浇鍣�
from models.resnet50 import get_resnet50
from models.vgg16 import get_vgg16
from models.efficientnet import get_efficientnet_b0
from models.densenet import get_densenet121
from models.mobilenet import get_mobilenet_v2

# 璁惧畾 CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 璁剧疆鏃ュ織
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'explanation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def load_model(model_name):
    if model_name == "ResNet50":
        model = get_resnet50(pretrained=True).to(device)
        target_layer = model.layer4[-1]
    elif model_name == "VGG16":
        model = get_vgg16(pretrained=True).to(device)
        target_layer = model.features[-1]
    elif model_name == "EfficientNet":
        model = get_efficientnet_b0(pretrained=True).to(device)
        target_layer = model.features[-1]
    elif model_name == "DenseNet":
        model = get_densenet121(pretrained=True).to(device)
        target_layer = model.features[-1]
    elif model_name == "MobileNet":
        model = get_mobilenet_v2(pretrained=True).to(device)
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, target_layer

def visualize_result(visualization, filename, method_name):
    """
    鍙鍖栬В閲婄粨鏋滃苟灞曠ず
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(visualization)
    plt.title(f"{method_name} for {filename}")
    plt.axis("off")
    plt.show()
    logging.info(f"Visualized {method_name} result for {filename}")

def save_result(visualization, filename, method_name):
    """
    淇濆瓨瑙ｉ噴缁撴灉鍒版湰鍦�
    """
    # 纭繚杈撳嚭鐩綍瀛樺湪
    output_dir = os.path.join("output", method_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{filename}_{method_name.lower()}.png")
    plt.imsave(output_path, visualization)
    logging.info(f"Saved {method_name} result to {output_path}")

def process_image_with_explainer(model, target_layer, image_pil, method_name, preprocess_func):
    """
    使用指定的解释器处理图像
    """
    try:
        logging.info(f"Running {method_name}...")
        start_time = time.time()
        
        # 将PIL图像转换为numpy数组
        rgb_image = np.array(image_pil)
        
        if method_name == "Grad-CAM":
            visualization = run_grad_cam(model, target_layer, None, rgb_image)
        elif method_name == "Score-CAM":
            visualization = run_score_cam(model, target_layer, None, rgb_image)
        elif method_name == "Layer-CAM":
            visualization = run_layer_cam(model, target_layer, None, rgb_image)
        elif method_name == "LIME":
            visualization = lime_explanation(model, image_pil, preprocess_func)
        elif method_name == "SHAP":
            visualization = shap_explanation(model, image_pil, preprocess_func)
        else:
            raise ValueError(f"Unknown explanation method: {method_name}")
        
        logging.info(f"{method_name} completed in {time.time() - start_time:.2f} seconds")
        return visualization, True
    except Exception as e:
        logging.error(f"Error in {method_name}: {str(e)}")
        return None, False

def load_labels(labels_path):
    """
    加载CIFAR-10标签
    """
    try:
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        logging.info(f"Loaded {len(labels)} labels from {labels_path}")
        return labels
    except Exception as e:
        logging.error(f"Error loading labels: {str(e)}")
        return None

def get_image_paths(train_dir, test_dir):
    """
    获取训练集和测试集的图片路径
    """
    train_images = []
    test_images = []
    
    # 获取训练集图片
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.png'):
                train_images.append(os.path.join(root, file))
    
    # 获取测试集图片
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.png'):
                test_images.append(os.path.join(root, file))
    
    logging.info(f"Found {len(train_images)} training images and {len(test_images)} test images")
    return train_images, test_images

def compare_models(models_to_compare, image_pil):
    """
    比较不同模型对同一张图片的预测结果
    """
    results = []
    
    for model_name in models_to_compare:
        try:
            # 加载模型
            model, target_layer = load_model(model_name)
            model.eval()
            
            # 预处理图像
            input_tensor = preprocess_for_cam(image_pil).unsqueeze(0).to(device)
            
            # 获取预测结果
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class = torch.max(probabilities, 1)
                
            # 记录结果
            results.append({
                'model': model_name,
                'predicted_class': top_class.item(),
                'confidence': top_prob.item(),
                'all_probabilities': probabilities.cpu().numpy()[0]
            })
            
            logging.info(f"{model_name} prediction completed")
            
        except Exception as e:
            logging.error(f"Error in {model_name} prediction: {str(e)}")
            results.append({
                'model': model_name,
                'error': str(e)
            })
    
    return results

def generate_comparison_report(results, filename):
    """
    生成模型比较报告
    """
    report = f"Comparison Report for {filename}\n"
    report += "=" * 50 + "\n\n"
    
    # 创建表格数据
    table_data = []
    headers = ["Model", "Predicted Class", "Confidence", "Top 3 Classes"]
    
    for result in results:
        if 'error' in result:
            table_data.append([
                result['model'],
                "Error",
                "N/A",
                result['error']
            ])
        else:
            # 获取前3个最可能的类别
            top3_indices = np.argsort(result['all_probabilities'])[-3:][::-1]
            top3_classes = [f"Class {idx} ({result['all_probabilities'][idx]:.4f})" for idx in top3_indices]
            
            table_data.append([
                result['model'],
                result['predicted_class'],
                f"{result['confidence']:.4f}",
                "\n".join(top3_classes)
            ])
    
    # 使用tabulate生成表格
    report += tabulate(table_data, headers=headers, tablefmt="grid")
    report += "\n\n"
    
    # 添加模型一致性分析
    if all('error' not in r for r in results):
        predictions = [r['predicted_class'] for r in results]
        if len(set(predictions)) == 1:
            report += "All models agree on the prediction.\n"
        else:
            report += "Models have different predictions:\n"
            for model, pred in zip([r['model'] for r in results], predictions):
                report += f"{model}: Class {pred}\n"
    
    return report

def compare_explainers(model, image_path, target_layer, use_cuda=True):
    """
    比较不同解释器的效果
    """
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        rgb_image = np.array(image)
        
        # 运行不同解释器
        results = {}
        
        # Grad-CAM
        grad_cam_result = run_grad_cam(model, target_layer, image, rgb_image, use_cuda)
        results['Grad-CAM'] = grad_cam_result
        
        # Score-CAM
        score_cam_result = run_score_cam(model, target_layer, image, rgb_image, use_cuda)
        results['Score-CAM'] = score_cam_result
        
        # Layer-CAM
        layer_cam_result = run_layer_cam(model, target_layer, image, rgb_image, use_cuda)
        results['Layer-CAM'] = layer_cam_result
        
        # LIME
        lime_result = lime_explanation(model, image, rgb_image, use_cuda)
        results['LIME'] = lime_result
        
        # SHAP
        shap_result = shap_explanation(model, image, rgb_image, use_cuda)
        results['SHAP'] = shap_result
        
        # 生成比较报告
        generate_comparison_report(results, image_path)
        
        # 可视化比较结果
        plot_comparison_results(results, image_path)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in compare_explainers: {str(e)}")
        raise

def plot_comparison_results(results, image_path):
    """
    可视化比较结果
    """
    try:
        # 创建输出目录
        output_dir = os.path.join("output", "visualizations", "comparison")
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备可视化数据
        titles = list(results.keys())
        visualizations = list(results.values())
        
        # 生成比较图
        save_path = os.path.join(output_dir, f"explainer_comparison_{os.path.basename(image_path)}.png")
        plot_comparison(visualizations, titles, save_path)
        
        logging.info(f"Comparison visualization saved to: {save_path}")
        
    except Exception as e:
        logging.error(f"Error in plot_comparison_results: {str(e)}")
        raise

def main():
    # 设置路径
    train_dir = r"E:\machinelearning\datacollection\CIFAR-10-100(含png图)\cifar_png\cifar\train"
    test_dir = r"E:\machinelearning\datacollection\CIFAR-10-100(含png图)\cifar_png\cifar\test"
    labels_path = r"E:\machinelearning\datacollection\CIFAR-10-100(含png图)\cifar_png\cifar\labels.txt"
    
    # 创建输出目录
    create_output_directories()
    
    # 设置日志
    log_file = setup_logging()
    logging.info(f"Log file created at: {log_file}")
    
    # 获取设备信息
    device_info = get_device_info()
    
    # 加载标签
    labels = load_labels(labels_path)
    if not labels:
        logging.error("Failed to load labels, exiting...")
        return
    
    # 获取图片路径
    train_images, test_images = get_image_paths(train_dir, test_dir)
    
    # 设置模型和图片路径
    models_to_compare = ["ResNet50", "VGG16", "EfficientNet", "DenseNet", "MobileNet"]
    
    # 统计信息
    stats = {
        "total_images": len(train_images) + len(test_images),
        "processed_images": 0,
        "successful_explanations": {method: 0 for method in ["Grad-CAM", "Score-CAM", "Layer-CAM", "LIME", "SHAP"]},
        "failed_explanations": {method: 0 for method in ["Grad-CAM", "Score-CAM", "Layer-CAM", "LIME", "SHAP"]},
        "model_comparisons": 0
    }
    
    # 处理训练集和测试集
    for image_set, image_paths in [("train", train_images), ("test", test_images)]:
        logging.info(f"\nProcessing {image_set} set...")
        
        for idx, image_path in enumerate(image_paths, 1):
            try:
                image_pil = Image.open(image_path).convert("RGB")
                filename = os.path.basename(image_path)
                
                logging.info(f"\nProcessing {image_set} image {idx}/{len(image_paths)}: {filename}")
                
                # 模型比较
                logging.info("Running model comparison...")
                comparison_results = compare_models(models_to_compare, image_pil)
                comparison_report = generate_comparison_report(comparison_results, filename)
                logging.info(comparison_report)
                
                # 保存比较报告
                report_dir = os.path.join("output", "comparison_reports", image_set)
                os.makedirs(report_dir, exist_ok=True)
                with open(os.path.join(report_dir, f"{filename}_comparison.txt"), "w") as f:
                    f.write(comparison_report)
                
                stats["model_comparisons"] += 1
                
                # 对每种解释方法进行处理
                for method_name in ["Grad-CAM", "Score-CAM", "Layer-CAM", "LIME", "SHAP"]:
                    preprocess_func = {
                        "Grad-CAM": preprocess_for_cam,
                        "Score-CAM": preprocess_for_cam,
                        "Layer-CAM": preprocess_for_cam,
                        "LIME": preprocess_for_lime,
                        "SHAP": preprocess_for_shap
                    }[method_name]
                    
                    # 对每个模型运行解释器
                    for model_name in models_to_compare:
                        try:
                            model, target_layer = load_model(model_name)
                            model.eval()
                            
                            logging.info(f"Running {method_name} with {model_name}...")
                            visualization, success = process_image_with_explainer(
                                model, target_layer, image_pil, method_name, preprocess_func
                            )
                            
                            if success and visualization is not None:
                                # 保存可视化结果
                                output_dir = os.path.join("output", "visualizations", method_name, model_name, image_set)
                                os.makedirs(output_dir, exist_ok=True)
                                save_path = os.path.join(output_dir, f"{filename}_{method_name.lower()}.png")
                                plt.imsave(save_path, visualization)
                                logging.info(f"Saved {method_name} visualization to: {save_path}")
                                
                                stats["successful_explanations"][method_name] += 1
                            else:
                                stats["failed_explanations"][method_name] += 1
                                
                        except Exception as e:
                            logging.error(f"Error processing {method_name} with {model_name}: {str(e)}")
                            stats["failed_explanations"][method_name] += 1
                
                stats["processed_images"] += 1
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                continue
    
    # 输出最终统计信息
    logging.info("\nFinal Statistics:")
    logging.info(f"Total images processed: {stats['total_images']}")
    logging.info(f"Successfully processed images: {stats['processed_images']}")
    logging.info(f"Model comparisons performed: {stats['model_comparisons']}")
    
    logging.info("\nExplanation Method Statistics:")
    for method in ["Grad-CAM", "Score-CAM", "Layer-CAM", "LIME", "SHAP"]:
        logging.info(f"{method}:")
        logging.info(f"  Successful: {stats['successful_explanations'][method]}")
        logging.info(f"  Failed: {stats['failed_explanations'][method]}")
    
    # 生成并保存最终报告
    final_report = f"""
Final Report
============
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Images: {stats['total_images']}
Processed Images: {stats['processed_images']}
Model Comparisons: {stats['model_comparisons']}

Explanation Method Performance:
-----------------------------
"""
    for method in ["Grad-CAM", "Score-CAM", "Layer-CAM", "LIME", "SHAP"]:
        success_rate = stats['successful_explanations'][method] / (stats['successful_explanations'][method] + stats['failed_explanations'][method]) * 100
        final_report += f"{method}:\n"
        final_report += f"  Successful: {stats['successful_explanations'][method]}\n"
        final_report += f"  Failed: {stats['failed_explanations'][method]}\n"
        final_report += f"  Success Rate: {success_rate:.2f}%\n\n"
    
    with open("output/final_report.txt", "w") as f:
        f.write(final_report)
    
    logging.info("Processing completed. Final report saved to output/final_report.txt")

if __name__ == "__main__":
    main()
