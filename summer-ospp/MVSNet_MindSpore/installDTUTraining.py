#!/usr/bin/python
# -*- coding:utf-8 -*-
from obs import ObsClient
import os
# pip install esdk-obs-python
AK = 'your ak'
SK = 'your sk'
server = 'your server'
bucketName = 'dtu'  # moxing路径中obs://dtu/ 的 "dtu"
obs_folder = 'dtu_training'  # moxing路径中 "dtu_training"
local_folder = '/home/ma-user/work/dtu_training'  

def download_obs_folder():
    obs_client = ObsClient(
        access_key_id=AK,
        secret_access_key=SK,
        server=server
    )
    try:
        os.makedirs(local_folder, exist_ok=True)
        resp = obs_client.listObjects(bucketName, prefix=obs_folder+'/')
        if resp.status < 300:
            print(f'开始下载OBS文件夹: obs://{bucketName}/{obs_folder}/ 到 {local_folder}')
            
            for content in resp.body.contents:
                object_key = content.key
                if object_key.endswith('/'):
                    continue
                relative_path = object_key[len(obs_folder)+1:]
                local_file_path = os.path.join(local_folder, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
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
        if 'obs_client' in locals():
            obs_client.close()

if __name__ == '__main__':
    download_obs_folder()
