import {ApiBuilder} from "@/assets/js/api/basic.js";

const devBaseURL = 'http://localhost:8100'
const prodBaseURL = 'https://project-api.coderjiang.com/ship-wise/python-api'
const baseURL = process.env.NODE_ENV === 'production' ? prodBaseURL : devBaseURL

const api = new ApiBuilder(baseURL)

const response_on_error = (err) => {
    console.error('请求失败，请检查网络连接或联系管理员。错误原因：' + err.message);
}

export const getVesselNum = async () => {
    try {
        const res = await api.get('/points/getVesselNum');
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}

export const getSpeed = async () => {
    try {
        const res = await api.get('/points/getSpeed');
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}

export const getVesselType = async () => {
    try {
        const res = await api.get('/points/getVesselType');
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}

export const getCountryNum = async () => {
    try {
        const res = await api.get('/points/getCountryNum');
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}

export const getLatestPos = async () => {
    try {
        const res = await api.get('/points/getLatestPos');
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}