<template>
	<div>
		<el-container>
			
			<el-container>
				<!-- 侧面功能栏 -->
				<el-aside>
					<div class="aside-box">
						<el-select v-model="compressAlg" placeholder="请选择压缩算法" style="margin-bottom: 40px;">
							<el-option-group
								v-for="group in options"
								:key="group.label"
								:label="group.label">
								<el-option
									v-for="item in group.options"
									:key="item.value"
									:label="item.label"
									:value="item.value">
								</el-option>
							</el-option-group>
						</el-select>
						<div class="param-box">
							<span style="margin-right: 10px;">阈值</span><el-tag>{{thrParam}}</el-tag>
							<el-slider v-model="thrParam" style="margin-left: 20px;margin-right: 10px;"></el-slider>
						</div>
						<div class="param-box">
							<span style="margin-right: 10px;">轨迹数量</span><el-tag>{{trajNum}}</el-tag>
							<el-slider v-model="trajNum" :step="10" style="margin-left: 20px;margin-right: 10px;"></el-slider>
						</div>
						<el-button type="primary" @click="startCompress">开始</el-button>
						<el-button type="danger" @click="reset">重置</el-button>
					</div>
					
				</el-aside>
				<!-- 中间主体部分 -->
				<el-main>
					<div class="main-box">
						<div id="map2"></div>
					</div>
				</el-main>
				
			</el-container>
			<!-- 底部栏 -->
			<el-footer style="background-color: #201f25;text-align: center;">
				<span class="span-bottom">
					压缩比：{{compressRate}}
				</span>
				<span class="span-bottom" style="margin-left: 50px;">
					耗时：{{useTime}}
				</span>
				<span class="span-bottom" style="margin-left: 50px;">
					其他描述
				</span>
			</el-footer>
		</el-container>
	</div>
</template>

<script>
export default {
	name: 'Compress',
	data() {
		return {
			imgUrl1: require('../../assets/address.png'),
			imgUrl2: require('../../assets/jiedao.png'),
			preList: [
				require('../../assets/address.png'),
				require('../../assets/jiedao.png')
			],
			options: [{
				label: "在线压缩算法",
				options: [{
					value: "OPW",
					label: "OPW"
				}, {
					value: "OPW-TR",
					label: "OPW-TR"
				}, {
					value: "Threshould",
					label: "Threshould"
				}, {
					value: "STTrace",
					label: "STTrace"
				}, {
					value: "SQUISH",
					label: "SQUISH"
				}, {
					value: "SQUISH-E",
					label: "SQUISH-E"
				}, {
					value: "Dead_Reckoning",
					label: "Dead_Reckoning"
				}]
			}, {
				label: "离线压缩算法",
				options: [{
					value: "Uniform",
					label: "Uniform"
				}, {
					value: "DP",
					label: "DP"
				}, {
					value: "TD-TR",
					label: "TD-TR"
				}]
			}],
			thrParam: 0,    //阈值
			trajNum: 0,     //轨迹数量
			compressAlg: "",    //压缩算法
			compressRate: 0,    //压缩比
			useTime: 0,     //耗时
			map: "",      //地图
			tianditu_1_tile: "",
			tianditu_1_marker: "",
			basicLayer: "",
			markLayer: "",
		}
	},
	methods: {
		//开始轨迹压缩
		startCompress() {
			
		},
		//重置
		reset() {
			this.thrParam = 0;
			this.trajNum = 0;
			this.compressAlg = "";
		},
		//初始化地图
		initMap() {
			//初始化地图对象
			this.map = L.map("map2", {
				center: [17.385044, 78.486671],    //地图中心
				zoom: 5,    //缩放比例
				zoomControl: false,     //禁用 + - 按钮
				doubleClickZoom: false,    //禁用双击放大
				attributionControl: false,     //移除右下角Leaflet标识  
			});
			//初始化地图图层（墨卡托坐标）
			this.tianditu_1_tile = "http://t4.tianditu.gov.cn/DataServer?T=vec_w&x={x}&y={y}&l={z}&tk=efe5e7d29b0c9728d44edba8dc08c8d7";
			this.tianditu_1_marker = "http://t4.tianditu.gov.cn/DataServer?T=cva_w&x={x}&y={y}&l={z}&tk=efe5e7d29b0c9728d44edba8dc08c8d7";
			//天地图影像图底图（墨卡托坐标）
			this.basicLayer = L.tileLayer(
				this.tianditu_1_tile,
			).addTo(this.map);
			
			//天地图影像图标记（墨卡托坐标）
			this.markLayer = L.tileLayer(
				this.tianditu_1_marker,
			).addTo(this.map);
		},
		//绘制轨迹
		drawTraj() {
			var latlngs = [
			    [[17.385044, 78.486671], [16.506174, 80.648015], [17.686816, 83.218482]],
			    [[13.082680, 80.270718], [12.971599, 77.594563],[15.828126, 78.037279]]
			];
			var multiPolyLineOptions = {color: 'red'};
			var multiPolyline = L.polyline(latlngs, multiPolyLineOptions);
			multiPolyline.addTo(this.map).on('click', function(e){
				console.log(e)
			});
		}
	},
	mounted() {
		this.initMap();
		this.drawTraj();
	}
}
</script>

<style lang="scss" scoped>
.span-bottom {
	font-size: 20px;
	color: white;
	line-height: 60px;
}
.el-footer {
	bottom: 0;
	position: absolute;
	width: 100%;
}
.aside-box {
	width: 100%;
	// height: 20px;
	// background-color: black;
	margin: 240px auto;
	text-align: center;
}
.el-button {
	width: 40%;
}
.param-box {
	margin-bottom: 25px;
}
.main-box {
	width: 100%;
	height: 100%;
}
</style>
