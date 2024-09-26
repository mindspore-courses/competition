<script>
	//定义公共属性和方法
	// const httpUrl = "http://127.0.0.1:8000/";
	const httpUrl = "http://127.0.0.1:4523/m1/5047827-0-default/";
	//转换船舶类型
	function transVesselType(num) {
		var vMap = {};
		vMap[9] = "渔船";
		vMap[10] = "客船";
		vMap[11] = "普通客船";
		vMap[12] = "旅游客船";
		vMap[13] = "高速客船";
		vMap[20] = "客货船";
		vMap[21] = "普通客货船";
		vMap[22] = "客货滚装船";
		vMap[30] = "普通货船";
		vMap[31] = "杂货船";
		vMap[32] = "散货船";
		vMap[33] = "滚装船";
		vMap[34] = "驳船";
		vMap[35] = "其他普通货船";
		vMap[40] = "集装箱船";
		vMap[50] = "液货船";
		vMap[51] = "油船";
		vMap[52] = "散装化学品船";
		vMap[53] = "液化气船";
		vMap[54] = "散装沥青船";
		vMap[55] = "油驳船";
		vMap[56] = "其他液货船";
		vMap[60] = "拖船";
		vMap[62] = "推轮";
		vMap[70] = "渡船";
		vMap[71] = "客渡船";
		vMap[72] = "车客渡船";
		vMap[73] = "汽车渡船";
		vMap[74] = "其他渡船";
		return vMap[num] == undefined ? "其他" : vMap[num];
	}
	//转换国旗
	function transCountry(countryName) {
		var flagUrl = {};
		flagUrl["中国"] = "http://api.shipxy.com/apiresource/flags/CHN.png";
		return flagUrl[countryName] == undefined ? "" : flagUrl[countryName];
	}
	export default {
		httpUrl,
		transVesselType,
		transCountry
	}
</script>

