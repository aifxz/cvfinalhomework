import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_for_lime  # 你自己写的预处理函数（如resize + to numpy + normalize）

def lime_explanation(model, image_pil, preprocess_func):
    # 将 PIL 图像转成 RGB 数组（H, W, 3）格式
    rgb_image = np.array(image_pil)

    explainer = lime_image.LimeImageExplainer()

    def predict(images_np_batch):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = torch.stack([
            preprocess_func(Image.fromarray(img.astype(np.uint8))) for img in images_np_batch
        ]).to(device)
        with torch.no_grad():
            outputs = model(batch)
        return outputs.detach().cpu().numpy()

    explanation = explainer.explain_instance(
        rgb_image,
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # 可视化
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=10,
        min_weight=0.0
    )

    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title("LIME Explanation")
    plt.axis('off')
    plt.show()
