import {ApiBuilder} from "@/assets/js/api/basic.js";

const devBaseURL = 'http://localhost:8100'
const prodBaseURL = 'https://project-api.coderjiang.com/ship-wise/python-api'
const baseURL = process.env.NODE_ENV === 'production' ? prodBaseURL : devBaseURL

const api = new ApiBuilder(baseURL)

const response_on_error = (err) => {
    console.error('请求失败，请检查网络连接或联系管理员。错误原因：' + err.message);
}

export const searchPoint = async (pic_name, mmsi) => {
    try {
        const res = await api.get('/points/searchPoint', {pic_name, mmsi});
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}

export const getDetections = async (startTime, endTime, picName) => {
    try {
        const res = await api.get('/detection/getDetections', {stime: startTime, etime: endTime, pic_name: picName});
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}

export const getSizes = async (picName) => {
    try {
        const res = await api.get('/detection/getSizes', {pic_name: picName});
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}

export const getTifInfo = async (picName) => {
    try {
        const res = await api.get('/points/getTifInfo', {pic_name: picName});
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}

export const getPoints = async (leftdown_lng, leftdown_lat, rightup_lng, rightup_lat) => {
    try {
        const res = await api.get('/points/getPoints', {leftdown_lng, leftdown_lat, rightup_lng, rightup_lat});
        return res.data;
    } catch (error) {
        response_on_error(error);
        throw error;
    }
}
