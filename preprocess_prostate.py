# 文件名: preprocess_prostate.py
import os
import SimpleITK as sitk
import numpy as np
from glob import glob
import cv2
import logging

def save_npy(data, p):
    """保存 .npy 文件，如果目录不存在则创建目录"""
    dir = os.path.dirname(p)
    os.makedirs(dir, exist_ok=True)
    np.save(p, data)

def normalize_slice(slice_data):
    """将切片标准化为零均值和单位方差"""
    slice_data = slice_data.astype(np.float32)
    mean = np.mean(slice_data)
    std = np.std(slice_data)
    if std > 0:
        return (slice_data - mean) / std
    else:
        return slice_data - mean # 避免除以零

def prepare_prostate(save_dir, ori_dir):
    """
    加载来自 6 个站点的 3D NIfTI 数据，并按 2D 切片进行预处理。
    基于 2007.02035v1.pdf 中的描述。
    
    假设 ori_dir 结构如下:
    - ori_dir/
        - SiteA/ (例如 RUNMC)
            - Case00.nii.gz
            - Case00_segmentation.nii.gz
            ...
        - SiteB/ (例如 BMC)
            ...
        - ... (直到 SiteF)
    """
    
    # 论文 中对应的站点名称
    # 您需要将 'SiteA' 到 'SiteF' 映射到您本地的实际文件夹名称
    site_names = ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']
    
    for i, site_name in enumerate(site_names):
        site_path = os.path.join(ori_dir, site_name)
        if not os.path.isdir(site_path):
            logging.warning(f"Directory not found for {site_name}, skipping.")
            continue
            
        # --- 修改点 1: 更新 glob 模式以匹配 *_segmentation.nii.gz ---
        seg_paths = glob(os.path.join(site_path, '*_segmentation.nii.gz'))
        seg_paths.sort()
        
        print(f'[INFO] Processing {site_name} (Site {i+1}) - Found {len(seg_paths)} cases.')

        for seg_path in seg_paths:
            
            # --- 修改点 2: 从分割路径推导图像路径 ---
            # 例如: Case00_segmentation.nii.gz -> Case00.nii.gz
            img_path = seg_path.replace('_segmentation.nii.gz', '.nii.gz') 
            
            if not os.path.exists(img_path):
                logging.warning(f"Image file not found for {seg_path} (Expected at {img_path}), skipping.")
                continue
            
            try:
                itk_image = sitk.ReadImage(img_path)
                itk_mask = sitk.ReadImage(seg_path)
                
                image = sitk.GetArrayFromImage(itk_image) # (Z, Y, X)
                mask = sitk.GetArrayFromImage(itk_mask)   # (Z, Y, X)
                
                # --- 修改点 3: 提取 Case_name ---
                case_name = os.path.basename(img_path).split('.')[0]

                # 论文 提到 "仅保留前列腺区域的切片"
                for k in range(image.shape[0]): # 遍历 Z 轴（轴向切片）
                    slice_mask = mask[k]
                    
                    # 仅保留包含前列腺的切片
                    if np.sum(slice_mask) > 0:
                        slice_image = image[k]
                        
                        # 1. 调整大小至 384x384
                        slice_image_resized = cv2.resize(slice_image, (384, 384), interpolation=cv2.INTER_LINEAR)
                        slice_mask_resized = cv2.resize(slice_mask, (384, 384), interpolation=cv2.INTER_NEAREST)
                        
                        # 2. 标准化 (零均值, 单位方差)
                        slice_image_norm = normalize_slice(slice_image_resized)
                        
                        # 3. 确保掩码是二值的
                        slice_mask_binary = (slice_mask_resized > 0).astype(np.uint8)
                        
                        base_name = f'{case_name}_slice_{k:03d}.npy'
                        
                        # 保存预处理后的 2D .npy 文件
                        save_npy(
                            slice_image_norm,
                            os.path.join(save_dir, 'prostate', f'Site{i+1}', 'image', base_name)
                        )
                        save_npy(
                            slice_mask_binary,
                            os.path.join(save_dir, 'prostate', f'Site{i+1}', 'mask', base_name)
                        )
                        
            except Exception as e:
                logging.error(f"Failed to process case {seg_path}: {e}")
                
    print("[INFO] Prostate preprocessing complete.")

if __name__ == '__main__':
    # --- 使用示例 ---
    logging.basicConfig(level=logging.INFO)
    
    # 1. 更改为您的前列腺数据集原始路径
    #    (假设它包含 SiteA, SiteB... 等子文件夹)
    ori_prostate_dir = '/path/to/your/raw_prostate_data' 
    
    # 2. 更改为您希望保存 .npy 文件的目标目录
    save_prostate_dir = '/path/to/your/preprocessed_prostate_data'

    # 运行预处理
    prepare_prostate(save_dir=save_prostate_dir, ori_dir=ori_prostate_dir)