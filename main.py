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
from utils.preprocess import preprocess_for_cam, preprocess_for_lime
from utils.helpers import create_output_directories, setup_logging, get_device_info
from utils.visualize import plot_comparison

# 瀵煎叆瑙ｉ噴鍣
from explainers.grad_cam import run_grad_cam
from explainers.score_cam import run_score_cam
from explainers.layer_cam import run_layer_cam
from explainers.lime_explainer import lime_explanation

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
            visualization = run_grad_cam(model, target_layer, image_pil, rgb_image)
        elif method_name == "Score-CAM":
            visualization = run_score_cam(model, target_layer, image_pil, rgb_image)
        elif method_name == "Layer-CAM":
            visualization = run_layer_cam(model, target_layer, image_pil, rgb_image)
        elif method_name == "LIME":
            visualization = lime_explanation(model, image_pil, rgb_image)
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
    """获取训练集和测试集的图片路径，每类采样相同数量"""
    train_images = []
    test_images = []
    
    # 每类采样数量
    samples_per_class_train = 80  # 训练集每类80张，共800张
    samples_per_class_test = 20   # 测试集每类20张，共200张
    
    # 获取训练集图片
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]
            # 随机采样
            sampled_images = np.random.choice(images, min(samples_per_class_train, len(images)), replace=False)
            train_images.extend(sampled_images)
    
    # 获取测试集图片
    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]
            # 随机采样
            sampled_images = np.random.choice(images, min(samples_per_class_test, len(images)), replace=False)
            test_images.extend(sampled_images)
    
    logging.info(f"Sampled {len(train_images)} training images and {len(test_images)} test images")
    return train_images, test_images

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
        lime_result = lime_explanation(model, image, rgb_image, use_cuda)
        results['LIME'] = lime_result
        
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
    # 设置日志
    setup_logging()
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 获取图片路径（采样1000张）
    train_images, test_images = get_image_paths("data/train", "data/test")
    
    # 加载模型
    models_to_compare = {
        "ResNet50": load_model("ResNet50"),
        "MobileNet": load_model("MobileNet")
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
            "LIME": process_image_with_explainer
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
            "LIME": process_image_with_explainer
        },
        os.path.join("output", "comparison_results", "test"),
        batch_size=50
    )
    
    # 生成最终报告
    generate_final_report(train_results, test_results)
    
    logging.info("All processing completed!")

if __name__ == "__main__":
    main()
