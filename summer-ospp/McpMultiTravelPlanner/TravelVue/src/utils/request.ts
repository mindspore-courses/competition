import axios from 'axios';
import type { AxiosResponse, AxiosRequestConfig } from 'axios';

// 创建axios实例
const request = axios.create({
  baseURL: "http://127.0.0.1:5000", // 从环境变量获取基础URL
  timeout: 60000, // 超时时间设置为60秒（因为涉及AI模型生成，可能耗时较长）
  headers: {
    'Content-Type': 'application/json'
  }
});

// 请求拦截器
request.interceptors.request.use(
  (config: import('axios').InternalAxiosRequestConfig) => {
    // 可以在这里添加认证信息，如token
    // const token = localStorage.getItem('token');
    // if (token && config.headers) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
request.interceptors.response.use(
  (response: AxiosResponse) => {
    // 直接返回响应数据中的data字段
    return response.data;
  },
  (error) => {
    // 统一错误处理
    let errorMessage = '请求失败，请稍后重试';
    
    if (error.response) {
      // 服务器返回错误
      switch (error.response.status) {
        case 400:
          errorMessage = error.response.data.error || '参数错误';
          break;
        case 500:
          errorMessage = error.response.data.error || '服务器内部错误';
          break;
        default:
          errorMessage = `请求错误: ${error.response.status}`;
      }
    } else if (error.request) {
      // 无响应
      errorMessage = '无法连接到服务器，请检查网络';
    }
    
    // 可以在这里添加全局错误提示，如使用Element Plus的Message
    // ElMessage.error(errorMessage);
    
    return Promise.reject(new Error(errorMessage));
  }
);

export default request;
