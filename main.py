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
from utils.preprocess import preprocess_for_cam
# from utils.preprocess import preprocess_for_lime  # 注释掉LIME预处理
from utils.helpers import create_output_directories, setup_logging, get_device_info
from utils.visualize import plot_comparison

# 瀵煎叆瑙ｉ噴鍣
from explainers.grad_cam import run_grad_cam
from explainers.score_cam import run_score_cam
from explainers.layer_cam import run_layer_cam
# from explainers.lime_explainer import lime_explanation  # 注释掉LIME解释器
# from explainers.shap_explainer import shap_explanation  # 注释掉SHAP解释器

# 瀵煎叆妯″瀷鍔犺浇鍣�
from models.resnet import ResNet50
from models.mobilenet import MobileNetV2

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
    """
    加载预训练模型
    Args:
        model_name: 模型名称
    Returns:
        tuple: (model, target_layer)
    """
    try:
        if model_name == "ResNet50":
            model = ResNet50(num_classes=10).to(device)
            target_layer = model.model.layer4[-1]
        elif model_name == "MobileNetV2":
            model = MobileNetV2(num_classes=10).to(device)
            target_layer = model.model.features[-1][-1].conv[0]
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 加载预训练权重
        model_path = os.path.join("output/models", f"{model_name.lower()}_cifar10.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.eval()
        logging.info(f"Loaded {model_name} model successfully")
        return model, target_layer
        
    except Exception as e:
        logging.error(f"Error loading {model_name} model: {str(e)}")
        raise

def create_output_directories():
    """创建必要的输出目录"""
    try:
        # 创建主输出目录
        os.makedirs("output", exist_ok=True)
        
        # 创建比较结果目录
        os.makedirs(os.path.join("output", "comparison_results", "train"), exist_ok=True)
        os.makedirs(os.path.join("output", "comparison_results", "test"), exist_ok=True)
        
        # 创建日志目录
        os.makedirs("output/logs", exist_ok=True)
        
        logging.info("Created all output directories successfully")
        
    except Exception as e:
        logging.error(f"Error creating output directories: {str(e)}")
        raise

def setup_logging():
    """设置日志配置"""
    try:
        # 创建日志目录
        os.makedirs("logs", exist_ok=True)
        
        # 设置日志文件名
        log_file = os.path.join("logs", f"explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"Logging setup completed. Log file: {log_file}")
        return log_file
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise

def get_device_info():
    """获取设备信息"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_info = {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        logging.info(f"Device information: {device_info}")
        return device_info
        
    except Exception as e:
        logging.error(f"Error getting device info: {str(e)}")
        raise

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

def process_image_with_explainer(image, model, explainer_name, explainer_func):
    """
    使用指定的解释器处理图像
    Args:
        image: PIL Image对象
        model: 预加载的模型
        explainer_name: 解释器名称
        explainer_func: 解释器函数
    Returns:
        dict: 包含热力图、预测结果和置信度的字典
    """
    try:
        logging.info(f"Running {explainer_name}...")
        start_time = time.time()
        
        # 将PIL图像转换为numpy数组
        rgb_image = np.array(image)
        
        # 根据解释器类型选择预处理函数
        if explainer_name in ["Grad-CAM", "Score-CAM", "Layer-CAM"]:
            preprocess_func = preprocess_for_cam
        # elif explainer_name == "LIME":  # 注释掉LIME
        #     preprocess_func = preprocess_for_lime
        # elif explainer_name == "SHAP":  # 注释掉SHAP
        #     preprocess_func = preprocess_for_shap
        else:
            raise ValueError(f"Unknown explanation method: {explainer_name}")
        
        # 获取模型的目标层
        target_layer = model[1]  # model是一个元组，包含模型和目标层
        
        # 运行解释器
        if explainer_name == "Grad-CAM":
            heatmap = run_grad_cam(model[0], target_layer, image, rgb_image)
        elif explainer_name == "Score-CAM":
            heatmap = run_score_cam(model[0], target_layer, image, rgb_image)
        elif explainer_name == "Layer-CAM":
            heatmap = run_layer_cam(model[0], target_layer, image, rgb_image)
        # elif explainer_name == "LIME":  # 注释掉LIME
        #     heatmap = lime_explanation(model[0], image, rgb_image)
        # elif explainer_name == "SHAP":  # 注释掉SHAP
        #     heatmap = shap_explanation(model[0], image, rgb_image)
        else:
            raise ValueError(f"Unknown explanation method: {explainer_name}")
        
        # 获取预测结果
        input_tensor = preprocess_func(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model[0](input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probabilities, 1)
        
        logging.info(f"{explainer_name} completed in {time.time() - start_time:.2f} seconds")
        
        return {
            'heatmap': heatmap,
            'prediction': top_class.item(),
            'confidence': top_prob.item(),
            'top_classes': probabilities.cpu().numpy()[0]
        }
        
    except Exception as e:
        logging.error(f"Error in {explainer_name}: {str(e)}")
        return None

def load_labels(labels_path):
    """
    加载CIFAR-10标签
    Args:
        labels_path: 标签文件路径
    Returns:
        list: 标签列表
    """
    try:
        if not os.path.exists(labels_path):
            logging.error(f"Labels file not found: {labels_path}")
            return None
            
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
            
        if not labels:
            logging.error("No labels found in the file")
            return None
            
        logging.info(f"Loaded {len(labels)} labels from {labels_path}")
        logging.info(f"Labels: {labels}")
        return labels
        
    except Exception as e:
        logging.error(f"Error loading labels: {str(e)}")
        return None

def get_image_paths(train_dir, test_dir):
    """获取训练集和测试集的图片路径，每类采样相同数量"""
    train_images = []
    test_images = []
    
    # 每类采样数量
    samples_per_class_train = 80  # 训练集每类80张，共800张
    samples_per_class_test = 20   # 测试集每类20张，共200张
    
    try:
        # 检查目录是否存在
        if not os.path.exists(train_dir):
            logging.error(f"Training directory not found: {train_dir}")
            return [], []
        if not os.path.exists(test_dir):
            logging.error(f"Test directory not found: {test_dir}")
            return [], []
            
        logging.info(f"Training directory: {train_dir}")
        logging.info(f"Test directory: {test_dir}")
        
        # 获取训练集图片
        train_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if train_files:
            # 随机采样
            sampled_images = np.random.choice(train_files, min(samples_per_class_train * 10, len(train_files)), replace=False)
            train_images = [os.path.join(train_dir, img) for img in sampled_images]
            logging.info(f"Sampled {len(train_images)} images from training set")
        else:
            logging.warning(f"No images found in training directory")
        
        # 获取测试集图片
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if test_files:
            # 随机采样
            sampled_images = np.random.choice(test_files, min(samples_per_class_test * 10, len(test_files)), replace=False)
            test_images = [os.path.join(test_dir, img) for img in sampled_images]
            logging.info(f"Sampled {len(test_images)} images from test set")
        else:
            logging.warning(f"No images found in test directory")
        
        logging.info(f"Total sampled images: {len(train_images)} training and {len(test_images)} test images")
        return train_images, test_images
        
    except Exception as e:
        logging.error(f"Error in get_image_paths: {str(e)}")
        raise

def compare_models(image_paths, models, explainers, output_dir, batch_size=50):
    """批量比较不同模型在相同图片上的解释结果"""
    results = []
    total_images = len(image_paths)
    processed_images = 0
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 分批处理图片
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(total_images-1)//batch_size + 1} ({len(batch_paths)} images)")
        
        for image_path in batch_paths:
            try:
                # 加载图片
                image = Image.open(image_path).convert('RGB')
                image_name = os.path.basename(image_path)
                
                # 为每个模型生成解释
                model_results = {}
                for model_name, model in models.items():
                    model_results[model_name] = {}
                    
                    # 使用不同的解释器
                    for explainer_name, explainer in explainers.items():
                        try:
                            # 生成解释
                            explanation = process_image_with_explainer(
                                image, 
                                model, 
                                explainer_name, 
                                explainer
                            )
                            
                            if explanation:
                                # 保存可视化结果
                                save_path = os.path.join(
                                    output_dir,
                                    f"{os.path.splitext(image_name)[0]}_{model_name}_{explainer_name}.png"
                                )
                                save_heatmap_overlay(
                                    image,
                                    explanation['heatmap'],
                                    save_path
                                )
                                
                                # 记录结果
                                model_results[model_name][explainer_name] = {
                                    'prediction': explanation['prediction'],
                                    'confidence': explanation['confidence'],
                                    'top_classes': explanation['top_classes']
                                }
                                
                        except Exception as e:
                            logging.error(f"Error processing {image_name} with {model_name} and {explainer_name}: {str(e)}")
                            continue
                
                # 记录所有模型的结果
                results.append({
                    'image': image_name,
                    'results': model_results
                })
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                continue
            
            processed_images += 1
            if processed_images % 10 == 0:
                logging.info(f"Processed {processed_images}/{total_images} images")
    
    # 保存比较结果
    save_comparison_report(results, os.path.join(output_dir, 'comparison_report.txt'))
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
        # lime_result = lime_explanation(model, image, rgb_image, use_cuda)
        # results['LIME'] = lime_result
        
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

def save_heatmap_overlay(image, heatmap, save_path):
    """
    保存热力图叠加结果
    Args:
        image: PIL Image对象
        heatmap: 热力图数组
        save_path: 保存路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 将PIL图像转换为numpy数组
        img_array = np.array(image)
        
        # 归一化热力图
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 创建热力图叠加
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_array)
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.title('Heatmap Overlay')
        plt.axis('off')
        
        # 保存结果
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Saved visualization to: {save_path}")
        
    except Exception as e:
        logging.error(f"Error saving heatmap overlay: {str(e)}")
        raise

def save_comparison_report(results, output_path):
    """
    保存比较报告
    Args:
        results: 比较结果列表
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w') as f:
            for result in results:
                f.write(f"\nImage: {result['image']}\n")
                f.write("=" * 50 + "\n")
                
                for model_name, model_results in result['results'].items():
                    f.write(f"\nModel: {model_name}\n")
                    f.write("-" * 30 + "\n")
                    
                    for explainer_name, explanation in model_results.items():
                        f.write(f"Explainer: {explainer_name}\n")
                        f.write(f"Prediction: {explanation['prediction']}\n")
                        f.write(f"Confidence: {explanation['confidence']:.4f}\n")
                        f.write("Top 3 Classes:\n")
                        top3_indices = np.argsort(explanation['top_classes'])[-3:][::-1]
                        for idx in top3_indices:
                            f.write(f"  Class {idx}: {explanation['top_classes'][idx]:.4f}\n")
                        f.write("\n")
                
                f.write("\n" + "=" * 50 + "\n")
        
        logging.info(f"Saved comparison report to: {output_path}")
        
    except Exception as e:
        logging.error(f"Error saving comparison report: {str(e)}")
        raise

def generate_final_report(train_results, test_results):
    """
    生成最终报告
    Args:
        train_results: 训练集结果
        test_results: 测试集结果
    """
    try:
        # 统计信息
        stats = {
            'total_images': len(train_results) + len(test_results),
            'train_images': len(train_results),
            'test_images': len(test_results),
            'successful_explanations': {
                'Grad-CAM': 0,
                'Score-CAM': 0,
                'Layer-CAM': 0
            },
            'failed_explanations': {
                'Grad-CAM': 0,
                'Score-CAM': 0,
                'Layer-CAM': 0
            }
        }
        
        # 统计成功和失败的解释
        for results in [train_results, test_results]:
            for result in results:
                for model_results in result['results'].values():
                    for explainer_name, explanation in model_results.items():
                        if explanation is not None:
                            stats['successful_explanations'][explainer_name] += 1
                        else:
                            stats['failed_explanations'][explainer_name] += 1
        
        # 生成报告
        report = f"""
Final Report
============
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Dataset Statistics:
-----------------
Total Images Processed: {stats['total_images']}
Training Images: {stats['train_images']}
Test Images: {stats['test_images']}

Explanation Method Performance:
-----------------------------
"""
        for method in ["Grad-CAM", "Score-CAM", "Layer-CAM"]:
            total = stats['successful_explanations'][method] + stats['failed_explanations'][method]
            success_rate = stats['successful_explanations'][method] / total * 100 if total > 0 else 0
            
            report += f"""
{method}:
  Successful: {stats['successful_explanations'][method]}
  Failed: {stats['failed_explanations'][method]}
  Success Rate: {success_rate:.2f}%
"""
        
        # 保存报告
        with open("output/final_report.txt", "w") as f:
            f.write(report)
        
        logging.info("Final report generated and saved to output/final_report.txt")
        
    except Exception as e:
        logging.error(f"Error generating final report: {str(e)}")
        raise

def main():
    try:
        # 设置日志
        log_file = setup_logging()
        logging.info(f"Log file created at: {log_file}")
        
        # 获取设备信息
        device_info = get_device_info()
        
        # 设置数据路径
        base_dir = r"E:\machinelearning\datacollection\CIFAR-10-100(含png图)\cifar_png\cifar"
        train_dir = os.path.join(base_dir, "train")
        test_dir = os.path.join(base_dir, "test")
        labels_path = os.path.join(base_dir, "labels.txt")
        
        # 检查路径是否存在
        if not os.path.exists(base_dir):
            logging.error(f"Base directory not found: {base_dir}")
            return
        if not os.path.exists(train_dir):
            logging.error(f"Training directory not found: {train_dir}")
            return
        if not os.path.exists(test_dir):
            logging.error(f"Test directory not found: {test_dir}")
            return
        if not os.path.exists(labels_path):
            logging.error(f"Labels file not found: {labels_path}")
            return
            
        logging.info(f"Base directory: {base_dir}")
        logging.info(f"Training directory: {train_dir}")
        logging.info(f"Test directory: {test_dir}")
        logging.info(f"Labels file: {labels_path}")
        
        # 创建输出目录
        create_output_directories()
        
        # 加载标签
        labels = load_labels(labels_path)
        if not labels:
            logging.error("Failed to load labels, exiting...")
            return
        
        # 获取图片路径（采样1000张）
        train_images, test_images = get_image_paths(train_dir, test_dir)
        if not train_images and not test_images:
            logging.error("No images found in dataset")
            return
            
        logging.info(f"Found {len(train_images)} training images and {len(test_images)} test images")
        
        # 加载模型
        models_to_compare = {
            "ResNet50": load_model("ResNet50"),
            "MobileNetV2": load_model("MobileNetV2")
        }
        
        # 处理训练集图片
        logging.info("Processing training images...")
        train_results = compare_models(
            train_images,
            models_to_compare,
            {
                "Grad-CAM": process_image_with_explainer,
                "Score-CAM": process_image_with_explainer,
                "Layer-CAM": process_image_with_explainer,
                # "LIME": process_image_with_explainer,  # 注释掉LIME
                # "SHAP": process_image_with_explainer,  # 注释掉SHAP
            },
            os.path.join("output", "comparison_results", "train"),
            batch_size=50
        )
        
        # 处理测试集图片
        logging.info("Processing test images...")
        test_results = compare_models(
            test_images,
            models_to_compare,
            {
                "Grad-CAM": process_image_with_explainer,
                "Score-CAM": process_image_with_explainer,
                "Layer-CAM": process_image_with_explainer,
                # "LIME": process_image_with_explainer,  # 注释掉LIME
                # "SHAP": process_image_with_explainer,  # 注释掉SHAP
            },
            os.path.join("output", "comparison_results", "test"),
            batch_size=50
        )
        
        # 生成最终报告
        generate_final_report(train_results, test_results)
        
        logging.info("All processing completed!")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
