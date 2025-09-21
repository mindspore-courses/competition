#!/usr/bin/python
# -*- coding:utf-8 -*-
from obs import ObsClient
import os
# pip install esdk-obs-python
# 配置信息（严格对齐moxing的成功路径）
AK = 'your ak'
SK = 'your sk'
server = 'your server'
bucketName = 'dtu'  # moxing路径中obs://dtu/ 的 "dtu"
obs_folder = 'dtu_training'  # moxing路径中 "dtu_training"（无末尾斜杠，与moxing一致）
local_folder = '/home/ma-user/work/dtu_training'  # 与moxing本地路径一致

def download_obs_folder():
    # 创建OBS客户端
    obs_client = ObsClient(
        access_key_id=AK,
        secret_access_key=SK,
        server=server
    )
    
    try:
        # 确保本地目录存在
        os.makedirs(local_folder, exist_ok=True)
        
        # 列出OBS文件夹中的所有对象
        resp = obs_client.listObjects(bucketName, prefix=obs_folder+'/')
        if resp.status < 300:
            print(f'开始下载OBS文件夹: obs://{bucketName}/{obs_folder}/ 到 {local_folder}')
            
            for content in resp.body.contents:
                # 获取对象键（相对路径）
                object_key = content.key
                
                # 跳过文件夹本身
                if object_key.endswith('/'):
                    continue
                
                # 构建本地文件路径
                relative_path = object_key[len(obs_folder)+1:]
                local_file_path = os.path.join(local_folder, relative_path)
                
                # 确保本地目录存在
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # 下载文件
                resp_get = obs_client.getObject(bucketName, object_key, downloadPath=local_file_path)
                if resp_get.status < 300:
                    print(f'下载成功: {object_key} -> {local_file_path}')
                else:
                    print(f'下载失败: {object_key}, 错误码: {resp_get.status}')
                    
            print('文件夹下载完成')
        else:
            print(f'无法列出OBS文件夹内容, 错误码: {resp.status}')
            
    except Exception as e:
        print(f'发生异常: {str(e)}')
    finally:
        # 关闭OBS客户端
        if 'obs_client' in locals():
            obs_client.close()

if __name__ == '__main__':
    download_obs_folder()
