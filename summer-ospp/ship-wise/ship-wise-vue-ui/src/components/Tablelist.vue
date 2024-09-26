<template>
	<div>
		<div style="line-height: 30px;float: center;">
			<h1><b>{{ date }}</b><br><b>{{ latlng }}</b>海域共<b>{{ ship_number }}</b>艘船</h1>
			<!-- <h1>船只统计信息：</h1> -->
		</div>
		<div class="echart" id="mychart" :style="myChartStyle"></div>
			<h1>
				<b>0-200(小型) 200-400(中型) 400以上(大型)</b>
			</h1>
	</div>
</template>

<script>
import * as echarts from "echarts";

export default {
	data() {
		return {
			myChart: null,
			// label1: [23, 24, 18, 25, 12], //数据
			// label2: [10, 13, 32, 42, 13],
			label3: [0, 0, 0],
			ship_number: 0,
			latlng: 0,
			date: null,
			myChartStyle: {float: "left", width: "350px", height: "340px" },//图表样式
		}
	},
	computed: {
		options() {
			return {
				// xAxis: {
				// 	// data: ["小型船只", "中型船只", "大型船只"]
				// },
				//图例
				legend: {
					data: ["小型船只", "中型船只", "大型船只"],
					// right:"10%",
					top: "10%",
					// orient:"vertical"
				},
				//鼠标划过时饼状图上显示的数据
				tooltip: {
					trigger: 'item',
					formatter: '{b}:{c} ({d}%)'
				},
				title: {
				// 设置饼图标题，位置设为顶部居中
				text: "船只统计信息",
				top: "0%",
				left: "center"
				},
				// yAxis: {},
				series: [
					{
						type: "pie", //形状为饼状图
						// data: this.label3,
						data: [
							{value:this.label3[0],name:"小型船只"},
							{value:this.label3[1],name:"中型船只"},
							{value:this.label3[2],name:"大型船只"},
							],// legend属性
						label: {
							// 饼状图上方文本标签，默认展示数值信息
							show: true,
							position: "inner",
							formatter: " {b}:{c} ({d}%)" // c代表对应值，d代表百分比
						},
						radius: "65%", //饼图半径
						avoidLabelOverlap: true, //是否启用防止标签重叠策略，默认开启
						
					},
				]
			}
		}
	},
	mounted() {
		this.initEcharts()
		this.$bus.on('ship_class', this.changeChart)
	},
	methods: {
		changeChart(shipclass, latlng, date) {
			let ship_number = 0
			let label3 = [shipclass.s, shipclass.m, shipclass.l]
			this.label3 = label3
			this.myChart.clear()
			this.myChart.setOption(this.options);
			for (var i = 0; i < 3; i++) {
				ship_number += label3[i]
			}
			this.ship_number = ship_number
			this.latlng = latlng
			this.date = date
		},
		//初始化echart
		initEcharts() {
			const myChart = echarts.init(document.getElementById("mychart"));
			this.myChart = myChart
			myChart.setOption(this.options);
			// //随着屏幕大小调节图表
			window.addEventListener("resize", () => {
				myChart.resize();
			});
		},
	}

}
</script>