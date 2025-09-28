import request from '../utils/request';

// 定义请求参数类型（与Flask后端接收的参数对应）
export interface TravelPlanParams {
  city: string;
  date: string;
  people: string; // 注意后端用的是peolpe(可能是拼写错误)，但前端传参仍用people
  tag: string | string[];
  activityIntensity: string;
  money: string;
  food: string;
}

// 定义单个行程项类型
export interface ItineraryItem {
  date: string;
  time_begin: string;
  time_end: string;
  name: string;
  spend_money: string;
  category: string;
  detail: string;
}

// 定义接口响应类型
export interface TravelPlanResponse {
  system_message: string;
  user_message: string;
  tool_message: string;
  response: string; // 这里实际是JSON数组字符串，后续需要解析为ItineraryItem[]
}

/**
 * 调用Flask后端生成旅游规划
 * @param params 旅游规划参数
 * @returns 规划结果
 */
export const getTravelResponse = async (params: TravelPlanParams): Promise<TravelPlanResponse> => {
  const res = await request.post<TravelPlanResponse>('/api/getRespone', params);
  return res.data;
};

/**
 * 解析响应中的行程数据为对象数组
 * @param responseStr 后端返回的response字段字符串
 * @returns 行程项数组
 */
export const parseItinerary = (responseStr: string): ItineraryItem[] => {
  try {
    // 处理可能的格式问题（比如后端返回的字符串可能包含多余的分号或空格）
    const cleaned = responseStr.replace(/;/g, '').trim();
    // 如果是数组形式但缺少外层括号，补充完整
    const jsonStr = cleaned.startsWith('[') ? cleaned : `[${cleaned}]`;
    return JSON.parse(jsonStr) as ItineraryItem[];
  } catch (error) {
    console.error('解析行程数据失败:', error);
    return [];
  }
};