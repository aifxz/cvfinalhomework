import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import logging
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from datetime import datetime
from tqdm import tqdm
from utils.metrics import calculate_entropy, calculate_sparsity, calculate_clarity, calculate_similarity, calculate_robustness
import matplotlib.patches as mpatches

# 设置日志
log_dir = "output/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_progress(message):
    """打印进度信息"""
    print(f"\r{message}", end="", flush=True)

def load_heatmap(heatmap_path):
    """加载热力图文件"""
    try:
        # 尝试加载原始热力图数据
        original_path = heatmap_path.replace('.png', '.npy')
        if os.path.exists(original_path):
            heatmap = np.load(original_path)
        else:
            # 如果原始数据不存在，从PNG文件加载
            heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
            if heatmap is None:
                raise ValueError(f"无法加载热力图: {heatmap_path}")
            heatmap = heatmap.astype(np.float32) / 255.0
        
        # 应用高斯平滑
        heatmap = gaussian_filter(heatmap, sigma=1)
        
        # 归一化
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        
        return heatmap
    except Exception as e:
        logging.error(f"加载热力图时出错 {heatmap_path}: {str(e)}")
        raise

def calculate_heatmap_quality(heatmap):
    """计算热力图质量指标"""
    # 计算熵
    hist = np.histogram(heatmap, bins=256, range=(0, 1))[0]
    hist = hist / hist.sum()
    heatmap_entropy = entropy(hist)
    
    # 计算稀疏性
    sparsity = np.mean(heatmap > 0.5)
    
    # 计算清晰度（边缘强度）
    sobelx = cv2.Sobel(heatmap, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(heatmap, cv2.CV_64F, 0, 1, ksize=3)
    clarity = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    return {
        'entropy': heatmap_entropy,
        'sparsity': sparsity,
        'clarity': clarity
    }

def calculate_explanation_consistency(heatmap1, heatmap2):
    """计算两个热力图之间的一致性"""
    # 计算余弦相似度
    flat1 = heatmap1.flatten()
    flat2 = heatmap2.flatten()
    similarity = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-10)
    
    # 计算鲁棒性（相对变化）
    diff = np.abs(heatmap1 - heatmap2)
    robustness = 1 - np.mean(diff)
    
    return {
        'similarity': similarity,
        'robustness': robustness
    }

def analyze_results():
    logging.info("Starting analysis...")
    
    # 定义解释器类型
    explainers = ['Grad-CAM', 'Layer-CAM', 'Occlusion', 'RandomMasking']
    
    # 初始化结果字典
    results = {
        'explainers': {explainer: {
            'entropy': [],
            'sparsity': [],
            'clarity': [],
            'similarity': [],
            'robustness': []
        } for explainer in explainers}
    }
    
    # 遍历所有热力图文件
    train_dir = "output/comparison_results2/test"
    if not os.path.exists(train_dir):
        logging.error(f"Directory {train_dir} does not exist")
        return
    
    # 获取所有热力图文件
    heatmap_files = [f for f in os.listdir(train_dir) if f.endswith('_original.npy')]
    logging.info(f"Found {len(heatmap_files)} heatmap files")
    
    # 分析每个热力图
    for heatmap_file in tqdm(heatmap_files, desc="Analyzing heatmaps"):
        try:
            # 解析文件名获取信息
            parts = heatmap_file.split('_')
            for part in parts:
                if part in ['Grad-CAM', 'Layer-CAM', 'Occlusion', 'RandomMasking']:
                    explainer = part
                    break
            else:
                continue
            
            # 加载热力图数据
            heatmap_path = os.path.join(train_dir, heatmap_file)
            heatmap = np.load(heatmap_path)
            
            # 计算各项指标
            entropy = calculate_entropy(heatmap)
            sparsity = calculate_sparsity(heatmap)
            clarity = calculate_clarity(heatmap)
            robustness = calculate_robustness(heatmap)
            
            # 存储结果
            results['explainers'][explainer]['entropy'].append(entropy)
            results['explainers'][explainer]['sparsity'].append(sparsity)
            results['explainers'][explainer]['clarity'].append(clarity)
            results['explainers'][explainer]['robustness'].append(robustness)
            
            # 计算与其他解释器的相似度
            for other_file in heatmap_files:
                if other_file != heatmap_file:
                    other_parts = other_file.split('_')
                    for part in other_parts:
                        if part in ['Grad-CAM', 'Layer-CAM', 'Occlusion', 'RandomMasking']:
                            other_explainer = part
                            break
                    else:
                        continue
                    
                    if other_explainer != explainer:
                        other_heatmap = np.load(os.path.join(train_dir, other_file))
                        similarity = calculate_similarity(heatmap, other_heatmap)
                        results['explainers'][explainer]['similarity'].append(similarity)
            
        except Exception as e:
            logging.error(f"Error processing {heatmap_file}: {str(e)}")
    
    # 计算平均指标
    for explainer in explainers:
        for metric in ['entropy', 'sparsity', 'clarity', 'similarity', 'robustness']:
            values = results['explainers'][explainer][metric]
            if values:
                mean = np.mean(values)
                std = np.std(values)
            else:
                mean = float('nan')
                std = float('nan')
            results['explainers'][explainer][metric] = {
                'mean': mean,
                'std': std
            }
    
    # 生成详细报告
    report_path = "output/comparison_results2/analysis_reports/overall_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Explainer Comparison Report\n")
        f.write("========================\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("------------------\n")
        
        # 创建表格数据
        table_data = []
        for explainer in explainers:
            row = [explainer]
            for metric in ['entropy', 'sparsity', 'clarity', 'similarity', 'robustness']:
                if isinstance(results['explainers'][explainer][metric], dict):
                    mean = results['explainers'][explainer][metric]['mean']
                    std = results['explainers'][explainer][metric]['std']
                    row.append(f"{mean:.4f} ± {std:.4f}")
            table_data.append(row)
        
        # 写入表格
        f.write(tabulate(table_data, 
                        headers=['Explainer', 'Entropy', 'Sparsity', 'Clarity', 'Similarity', 'Robustness'],
                        tablefmt='grid'))
        
        f.write("\n\nDetailed Analysis:\n")
        f.write("----------------\n")
        
        # 对每个指标进行详细分析
        for metric in ['entropy', 'sparsity', 'clarity', 'similarity', 'robustness']:
            f.write(f"\n{metric.upper()} Analysis:\n")
            values = {explainer: results['explainers'][explainer][metric]['mean'] 
                     for explainer in explainers}
            
            # 找出最佳和最差表现
            best_explainer = max(values.items(), key=lambda x: x[1])[0]
            worst_explainer = min(values.items(), key=lambda x: x[1])[0]
            
            f.write(f"Best performing: {best_explainer} ({values[best_explainer]:.4f})\n")
            f.write(f"Worst performing: {worst_explainer} ({values[worst_explainer]:.4f})\n")
            
            # 计算相对性能
            for explainer in explainers:
                if explainer != best_explainer:
                    relative_performance = values[explainer] / values[best_explainer]
                    f.write(f"{explainer} achieves {relative_performance:.2%} of {best_explainer}'s performance\n")
        
        f.write("\n\nInterpretation:\n")
        f.write("--------------\n")
        f.write("1. Entropy: Higher values indicate more diverse attention distribution\n")
        f.write("2. Sparsity: Higher values indicate more focused attention\n")
        f.write("3. Clarity: Higher values indicate clearer and more distinct attention patterns\n")
        f.write("4. Similarity: Higher values indicate more consistent explanations across different methods\n")
        f.write("5. Robustness: Higher values indicate more stable explanations under noise\n")
        
        f.write("\n\nKey Findings:\n")
        f.write("------------\n")
        
        # 总结关键发现
        f.write("1. Performance Comparison:\n")
        for metric in ['entropy', 'sparsity', 'clarity', 'similarity', 'robustness']:
            values = {explainer: results['explainers'][explainer][metric]['mean'] 
                     for explainer in explainers}
            best = max(values.items(), key=lambda x: x[1])
            f.write(f"   - {metric}: {best[0]} performs best with {best[1]:.4f}\n")
        
        f.write("\n2. Consistency Analysis:\n")
        for explainer in explainers:
            std_values = [results['explainers'][explainer][metric]['std'] 
                         for metric in ['entropy', 'sparsity', 'clarity', 'similarity', 'robustness']]
            avg_std = np.mean(std_values)
            f.write(f"   - {explainer} shows {avg_std:.4f} average standard deviation across metrics\n")
    
    logging.info(f"Analysis completed. Report saved to {report_path}")

    # ===== 可视化部分（升级版） =====
    # 莫兰迪配色
    morandi_colors = ['#A3A9AA', '#C1B7A4', '#B7AFA3', '#A3B7B0']
    short_names = ['Grad', 'Layer', 'Occ', 'RandMask']
    metrics = ['entropy', 'sparsity', 'clarity', 'similarity', 'robustness']
    explainer_names = explainers
    means = np.array([[results['explainers'][e][m]['mean'] for m in metrics] for e in explainer_names])
    stds = np.array([[results['explainers'][e][m]['std'] for m in metrics] for e in explainer_names])
    # 1. 单独画 entropy
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    x = np.arange(len(short_names))
    entropy_means = means[:, 0]
    entropy_stds = stds[:, 0]
    bars = []
    for i, (mean, std) in enumerate(zip(entropy_means, entropy_stds)):
        if np.isnan(mean):
            bar = ax1.bar(x[i], 0.01, color='gray', alpha=0.3, label='NaN' if i==0 else "")
            ax1.text(x[i], 0.02, '*', ha='center', va='bottom', fontsize=14, color='gray')
        else:
            bar = ax1.bar(x[i], mean, yerr=std, color=morandi_colors[i], capsize=4)
            ax1.text(x[i], mean+0.02, f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
        bars.append(bar)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=15)
    ax1.set_ylabel('Entropy')
    ax1.set_title('Explainer Entropy Comparison')
    handles = [mpatches.Patch(color=morandi_colors[i], label=short_names[i]) for i in range(len(short_names))]
    handles.append(mpatches.Patch(color='gray', alpha=0.3, label='NaN'))
    ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig("output/comparison_results2/analysis_reports/overall_entropy_barplot.png", bbox_inches='tight')
    plt.close()

    # 2. 其余四项合并画
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(short_names))
    width = 0.18
    for j, metric in enumerate(metrics[1:]):
        for i in range(len(short_names)):
            mean = means[i, j+1]
            std = stds[i, j+1]
            xpos = x[i] + (j-1.5)*width
            if np.isnan(mean):
                bar = ax2.bar(xpos, 0.01, width, color='gray', alpha=0.3, label='NaN' if (i==0 and j==0) else "")
                ax2.text(xpos, 0.02, '*', ha='center', va='bottom', fontsize=12, color='gray')
            else:
                bar = ax2.bar(xpos, mean, width, yerr=std, color=morandi_colors[i], capsize=4)
                ax2.text(xpos, mean+0.01, f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=15)
    ax2.set_ylabel('Score')
    ax2.set_title('Explainer Metrics Comparison (Sparsity, Clarity, Similarity, Robustness)')
    # 构建图例
    handles = [mpatches.Patch(color=morandi_colors[i], label=short_names[i]) for i in range(len(short_names))]
    handles.append(mpatches.Patch(color='gray', alpha=0.3, label='NaN'))
    ax2.legend(handles=handles, loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig("output/comparison_results2/analysis_reports/overall_metrics_barplot.png", bbox_inches='tight')
    plt.close()

    # 3. 结果分布箱线图（每个解释器每个指标的原始分布）
    fig3, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    for j, metric in enumerate(metrics):
        data = [results['explainers'][e][metric]['mean'] for e in explainer_names]
        axes[j].bar(short_names, data, color=morandi_colors)
        axes[j].set_title(metric)
        axes[j].set_ylabel('Score')
        for i, val in enumerate(data):
            if not np.isnan(val):
                axes[j].text(i, val+0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            else:
                axes[j].bar(i, 0.01, color='gray', alpha=0.3)
                axes[j].text(i, 0.02, '*', ha='center', va='bottom', fontsize=12, color='gray')
    plt.tight_layout()
    plt.savefig("output/comparison_results2/analysis_reports/overall_boxplot.png", bbox_inches='tight')
    plt.close()

    # 图注说明
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n注：灰色透明栏和*表示该方法在该指标下无有效输出（NaN），反映其在该维度下缺乏稳定可视解释能力。\n")

def generate_comparison_report(results, output_dir):
    """生成比较报告"""
    try:
        # 创建报告文件
        report_path = os.path.join(output_dir, "overall_report2.txt")
        with open(report_path, "w") as f:
            f.write("Overall Analysis Report\n")
            f.write("=====================\n\n")
            
            # 模型比较
            f.write("Model Comparison:\n")
            f.write("----------------\n")
            for model, metrics in results['model_comparison'].items():
                f.write(f"\n{model}:\n")
                for metric, values in metrics.items():
                    if values:  # 确保列表不为空
                        avg = np.mean(values)
                        std = np.std(values)
                        f.write(f"  {metric}: {avg:.4f} ± {std:.4f}\n")
            
            # 解释器比较
            f.write("\nExplainer Comparison:\n")
            f.write("-------------------\n")
            for method, metrics in results['explainer_comparison'].items():
                f.write(f"\n{method}:\n")
                for metric, values in metrics.items():
                    if values:  # 确保列表不为空
                        avg = np.mean(values)
                        std = np.std(values)
                        f.write(f"  {metric}: {avg:.4f} ± {std:.4f}\n")
            
            # 类别分析
            f.write("\nCategory Analysis:\n")
            f.write("----------------\n")
            for category, metrics in results['category_analysis'].items():
                f.write(f"\n{category}:\n")
                for metric, values in metrics.items():
                    if values:  # 确保列表不为空
                        avg = np.mean(values)
                        std = np.std(values)
                        f.write(f"  {metric}: {avg:.4f} ± {std:.4f}\n")
        
        logging.info(f"报告已保存到: {report_path}")
        
    except Exception as e:
        logging.error(f"生成报告时出错: {str(e)}")
        raise

if __name__ == "__main__":
    analyze_results() 