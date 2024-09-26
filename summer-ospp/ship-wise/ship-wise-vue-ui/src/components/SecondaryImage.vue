<template>  
	<div>
	<h1 style="color: #fff;font-size:18px;text-align: center;">船只预测</h1>
	<div class = "box1" id="id" style="width:350px;height:350px;">
<!-- 		<el-image 
		:src="imageurl"
		style="width: ;position:absolute; top:0px; left:0px; clip:rect(34px 135px 91px 1px);" 
		>
		</el-image> -->
	</div>
	<!-- <div class = "dcontainer" style="width:300px;height:150px;"> -->
	<div>
		<!-- <el-button type="primary" @click="changeImage()()">切换</el-button> -->
	</div>
		
	</div>
</template>
	
<script>
import * as echarts from "echarts";
export default {
	data() {
		return {
			imagenumber:0,
			//imageurl:require('../src/assets/ship1.jpg'),
			srr: require('../assets/ship2.jpg'),
			posx: "-100px",
			posy: "-100px"
		}
	},

	created(){
		this.$bus.on('ship_data2',this.changeImage);//获取点击蓝色标记时传递过来的子图路径/名称，和该子图内对应红框范围数组
	},
	methods: {
		
		//采用背景精灵图格式进行局部展示，缺点是原图大小不可以改变
		changeImage(path,boxary){ //path,boxary 子图名称或路径，红框范围数组：对应的的数值分别为上 左 下 右
			//let boxary = [0.0, 692.3465, 43.156414, 745.1226];
			//alert("heiheie");
			var test = document.getElementById('id');//获取对应css样式
			console.log("success",path,boxary)
			//动态修改背景图
			//this.srr = require('../Alldetectionimages/WC1-01_20220601_0000000001_0006_0060_E118.0_N19.4_L2A_PAN_5_24.jpg');
			this.srr =  'http://127.0.0.1:8000/detection/getSubPic/?sub_name=' + path;//去图片库里寻找子图
			test.style.backgroundImage = 'url("'+ this.srr+'")';
			console.log("数组值",boxary["top"],boxary["left"],boxary["bottom"],boxary["right"]);
			//修改移动坐标
			let x = (boxary["left"] + boxary["right"])/2 - 175; //减去box1局部框大小的一半
			let y = (boxary["top"] + boxary["bottom"])/2 - 175;
			//边界判断
			if(x <= 0){
				x = 0;
			}else if(x>=650){
				x = 650;
			}
			if(y <= 0){
				y = 0;
			}else if(y >= 650){
				y = 650;
			}
			this.posx = "-" + x + "px";
			this.posy = "-" + y + "px";
			test.style.backgroundPositionX = this.posx;
			test.style.backgroundPositionY = this.posy;	
		}
	},
}
</script>>



<style scoped>
::v-deep .box1 {
 	
	background-image: url('../assets/ship0.jpg');
 	background-position:  -100.2481px -200.2234px;
 	/** 使用var()获取变量，参数就是变量名 */
 /* 	  background-: var(--backgroundImage);
 	  color: var(--fontColor); */
 }
 
</style>

 
 
