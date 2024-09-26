<template>
  <div>
    <el-container>
      <!-- 标题栏 -->
      <el-header style="background-color: #18223d;height: calc(7vh);">
        <!-- <div class="visual_title_left">
          <span class="visual_title_span" @click="skipToBack"><i class="el-icon-d-arrow-left" style="margin-right: 5px;"></i>返回</span>
        </div> -->
        <div class="visual_title">
          <span class="visual_title_span" @click="skipToBack" style="float: left;"><i class="el-icon-d-arrow-left"
                                                                                      style="margin-right: 5px;"></i>返回</span>
          <span>船舶轨迹大数据可视化面板</span>
          <span class="visual_title_span" @click="skipToMain" style="float: right;">主页<i class="el-icon-s-home"
                                                                                           style="margin-left: 5px;"></i></span>
        </div>
        <!-- <div class="visual_title_right">
          <span class="visual_title_span" @click="skipToMain">主页<i class="el-icon-s-home" style="margin-left: 5px;"></i></span>
        </div> -->
      </el-header>
      <!-- 主体部分 -->
      <el-main style="padding: 0;">
        <!-- 左半部分 -->
        <div class="div_left">
          <div class="div_left_top">
            <div id="speedEcharts"></div>
          </div>
          <div class="div_left_middle">
            <div id="vesselTypeEcharts"></div>
          </div>
          <div class="div_left_bottom">
            <div id="countryEcharts"></div>
          </div>
        </div>
        <!-- 中间部分 -->
        <div class="div_middle">
          <!-- 地图 -->
          <div class="div_middle_top">
            <div id="map2"></div>
          </div>
          <div class="div_middle_bottom">
            <div class="content_sum">
              <div class="sum_title">
                当前船舶总数：
              </div>
              <div class="sum_content">{{ input1 }}</div>
              <div class="sum_title">
                当前停泊船舶总数：
              </div>
              <div class="sum_content">{{ input2 }}</div>
            </div>
          </div>
        </div>
        <!-- 右半部分 -->
        <div class="div_right">
          <div class="div_right_title">船舶最新更新数据</div>
          <table>
            <tr>
              <th>日期</th>
              <th>MMSI</th>
              <th>操作</th>
            </tr>
            <tbody>
            <tr v-for="(item, index) in tableData">
              <td width="180" class="td_date"><i class="el-icon-time"></i> {{ item.date }}</td>
              <td width="180" @click="readDetail(index)" class="td_tag">
                <el-tag size="medium">{{ item.mmsi }}</el-tag>
              </td>
              <td width="180">
                <el-button type="primary" size="mini" @click="readMap(index)">查看地图</el-button>
              </td>
            </tr>
            </tbody>
          </table>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>
import global from './Common.vue'     //引入全局文件 
import * as echarts from 'echarts';      //引入Echarts
import {getCountryNum, getLatestPos, getSpeed, getVesselNum, getVesselType} from "@/assets/js/api/visualization-api";

export default {
  name: 'Visualization',
  data() {
    return {
      input1: '',     //当前船舶总数
      input2: '',     //当前停泊船舶总数
      tableData: [{   //表格数据
        date: "",
        mmsi: "",
        lat: "",
        lon: "",
        speed: "",
        course: "",
        heading: ""
      }],
      map: '',     // 地图
    };
  },
  methods: {
    //获取当前船舶总数
    getVesselNum() {
      //获取数据
      let that = this;
      getVesselNum()
          .then(function (response) {
            that.input1 = response.data.curNum;
            that.input2 = response.data.stopNum;
          }.bind(this))
          .catch(function (error) {
            console.log(error);
          });
    },
    //获取航速分布折线图
    speedEcharts() {
      let speedEchart = echarts.init(document.getElementById("speedEcharts"));
      //加载动画
      speedEchart.showLoading();
      //获取数据
      getSpeed()
          .then(function (response) {
            speedEchart.hideLoading();    //关闭加载动画
            //设置数据
            let data = response.data;
            let seriesData = [];
            for (let i = 0; i < data.length; i++) {
              let temp = [];
              temp.push(data[i][0]);
              temp.push(data[i][1]);
              seriesData.push(temp);
            }
            //绘制折线图
            let option = {
              title: {
                text: '船舶速度统计折线图',
                left: 'center',
                textStyle: {
                  color: '#fff'
                }
              },
              tooltip: {
                trigger: 'axis',
                formatter: '{c}'
              },
              xAxis: {
                type: 'value',
                name: '航速',
                boundaryGap: false,
                axisLabel: {
                  color: '#fff'
                },
                axisLine: {
                  onZero: false,
                  lineStyle: {
                    color: '#fff'
                  }
                },
                axisPointer: {
                  lineStyle: {
                    color: '#fff',
                    type: 'dashed'
                  }
                },
              },
              yAxis: {
                type: 'value',
                name: '船舶数量',
                axisLabel: {
                  color: '#fff'
                },
                axisLine: {
                  onZero: false,
                  lineStyle: {
                    color: '#fff'
                  }
                },
              },
              series: {
                data: seriesData,
                type: 'line',
                areaStyle: {
                  color: 'rgba(24, 144, 255, 0.08)'
                },
              }
            };
            speedEchart.setOption(option);
          }).catch(function (error) {
        console.log(error);
      });
    },
    //获取船舶种类柱状图
    vesselTypeEcharts() {
      let vesselTypeEchart = echarts.init(document.getElementById("vesselTypeEcharts"));
      //加载动画
      vesselTypeEchart.showLoading();
      //获取数据
      getVesselType()
          .then(function (response) {
            vesselTypeEchart.hideLoading();     //关闭加载动画
            //设置数据
            let legendData = [];
            let seriesData = [];
            let data = response.data;
            for (let i = 0; i < data.length; i++) {
              legendData.push(global.transVesselType(data[i][0]));
              seriesData.push(data[i][1]);
            }
            //绘制柱状图
            let option = {
              title: {
                text: '船舶种类统计柱状图',
                left: 'center',
                textStyle: {
                  color: '#fff'
                }
              },
              tooltip: {
                trigger: 'axis',
                formatter: '{b} : {c}'
              },
              xAxis: {
                type: 'category',
                name: '类别',
                data: legendData,
                axisLabel: {
                  color: '#fff'
                },
                axisLine: {
                  onZero: false,
                  lineStyle: {
                    color: '#fff'
                  }
                },
              },
              yAxis: {
                type: 'value',
                name: "船舶数量",
                axisLabel: {
                  color: '#fff'
                },
                axisLine: {
                  onZero: false,
                  lineStyle: {
                    color: '#fff'
                  }
                },
              },
              series: [
                {
                  type: 'bar',
                  data: seriesData,
                  showBackground: true,
                  backgroundStyle: {
                    color: 'rgba(180,180,180,0.2)'
                  }
                }
              ]
            };
            vesselTypeEchart.setOption(option);
          }).catch(function (error) {
        console.log(error);
      });
    },
    //获取国家名称饼状图
    countryEcharts() {
      let countryEchart = echarts.init(document.getElementById("countryEcharts"));
      //加载动画
      countryEchart.showLoading();
      //获取数据
      getCountryNum()
          .then(function (response) {
            countryEchart.hideLoading();    //关闭加载动画
            //设置数据
            let legendData = [];
            let seriesData = [];
            let data = response.data;
            for (let i = 0; i < data.length; i++) {
              legendData.push(data[i][0]);
              let temp = {};
              temp.value = data[i][1];
              temp.name = data[i][0];
              seriesData.push(temp);
            }
            //绘制饼状图
            let option = {
              title: {
                text: '船舶所属国籍统计饼状图',
                left: 'center',
                textStyle: {
                  color: '#fff'
                }
              },
              tooltip: {
                trigger: 'item',
                formatter: '{b} : {c} ({d}%)'
              },
              legend: {
                orient: 'vertical',
                left: 'left',
                data: legendData,
                textStyle: {
                  color: '#fff'
                }
              },
              series: [
                {
                  name: '船舶所属国籍统计饼状图',
                  type: 'pie',
                  radius: '55%',
                  center: ['50%', '60%'],
                  data: seriesData,
                  emphasis: {
                    itemStyle: {
                      normal: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0,0,0,0.5)'
                      }
                    }
                  }
                }
              ]
            };
            countryEchart.setOption(option);
          }).catch(function (error) {
        console.log(error);
      });
    },
    //获取船舶最新位置
    getLatestPos() {
      //清空数组后重建
      this.tableData = undefined;
      this.tableData = [];
      //获取数据
      let that = this;
      getLatestPos()
          .then(function (response) {
            let data = response.data;
            for (let i = 0; i < data.length; i++) {
              that.tableData.push({
                date: data[i][0],
                mmsi: data[i][1],
                lat: data[i][2],
                lon: data[i][3],
                speed: data[i][4],
                course: data[i][5],
                heading: data[i][6]
              });
            }
          }.bind(this))
          .catch(function (error) {
            console.log(error);
          });
    },
    //获取实时数据详细内容
    readDetail(index) {
      let data = this.tableData;
      const h = this.$createElement;
      this.$notify({
        title: '实时数据内容',
        type: 'success',
        message: h('div', null, [
          h('p', null, "纬度：" + data[index].lat),
          h('p', null, '经度：' + data[index].lon),
          h('p', null, '航速：' + data[index].speed),
          h('p', null, '航向：' + data[index].course),
          h('p', null, '船首向：' + data[index].heading)
        ])
      });
    },
    //初始化地图
    initMap() {
      //初始化地图对象
      this.map = L.map("map2", {
        center: [40.02404009136253, 105.10641060224784],    //地图中心
        zoom: 4,    //缩放比例
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
    //在地图查看数据点
    readMap(index) {
      let data = this.tableData;
      let lat = data[index].lat;
      let lon = data[index].lon;
      //移动地图中心点
      this.map.setView([lat, lon], 5);
      //设置中心点图标
      let centerIcon = L.Icon.extend({
        options: {
          iconUrl: require('../assets/loc_red.png'),
          iconSize: [30, 30]
        }
      });
      L.marker([lat, lon], {
        icon: new centerIcon()
      }).bindPopup("<p>纬度：" + lat + "</p><p>经度：" + lon + "</p>").openPopup().addTo(this.map);
    },
    //返回上一页
    skipToBack() {
      window.history.go(-1);
    },
    //跳转到主页
    skipToMain() {
      this.$router.push('/');
    },
  },
  mounted() {
    let that = this;
    this.speedEcharts();
    this.getVesselNum();
    this.countryEcharts();
    this.vesselTypeEcharts();
    this.getLatestPos();
    this.initMap();
    //定时刷新
    setInterval(function () {
      that.speedEcharts()
    }, 60 * 1000);
    setInterval(function () {
      that.getVesselNum()
    }, 60 * 1000);
    setInterval(function () {
      that.countryEcharts()
    }, 60 * 1000);
    setInterval(function () {
      that.vesselTypeEcharts()
    }, 60 * 1000);
    setInterval(function () {
      that.getLatestPos()
    }, 60 * 1000);
  }
}
</script>

<style>
.visual_title {
  width: 100%;
  font-size: 22px;
  line-height: calc(7vh);
  text-align: center;
  color: #fff;
  margin: 0 auto;
}

.visual_title_span {
  font-size: 17px;
  padding-left: 8px;
  padding-right: 8px;
}

.visual_title_span:hover {
  cursor: pointer;
  background-color: rgba(255, 255, 255, 0.2) !important;
}

.div_left, .div_middle, .div_right {
  background: linear-gradient(to top, #00c996, #003d4d);
}

.div_left {
  width: 30%;
  height: calc(93vh);
  position: absolute;
  left: 0;
  top: calc(7vh);
}

.div_middle {
  width: 40%;
  height: calc(93vh);
  position: absolute;
  left: 30%;
}

.div_right {
  width: 30%;
  height: calc(93vh);
  position: absolute;
  right: 0;
  top: calc(7vh);
  border-left: solid #0078A8;
}

.div_left_top, .div_left_middle, .div_left_bottom {
  width: 100%;
  height: calc(31vh);
}

.div_middle_top {
  width: 100%;
  height: 80%;
}

.div_middle_bottom {
  width: 100%;
  height: 20%;
  border-left: solid #0078A8;
}

#speedEcharts {
  width: 100%;
  height: 100%;
}

#vesselTypeEcharts {
  width: 100%;
  height: 100%;
}

#countryEcharts {
  width: 100%;
  height: 100%;
}

.content_sum {
  font-size: 17px;
  color: #fff;
  padding-top: 10px;
  padding-bottom: 10px;
  padding-left: 15px;
  padding-right: 15px;
}

.sum_content {
  width: 80%;
  height: 20px;
  line-height: 20px;
  text-align: center;
  position: relative;
  margin: 0 auto;
  margin-top: 10px;
  margin-bottom: 5px;
  box-shadow: 0 2px 4px rgba(255, 255, 255, .80);
}

.div_right_title {
  width: 90%;
  margin: 0 auto;
  text-align: center;
  font-size: 21px;
  padding: 20px;
  padding-bottom: 5px;
  color: #fff;
  font-weight: 600;
}

.div_right table {
  border-collapse: collapse;
  border: none;
  width: 90%;
  margin: 0 auto;
  margin-top: 10px;
}

.div_right tbody tr:hover {
  background-color: rgba(0, 0, 0, 0.1);
}

.div_right td, .div_right th {
  text-align: left;
  border-bottom: 1px solid #ebeef5;
  padding: 10px;
}

.div_right tbody td {
  font-size: 10px;
}

.td_date {
  color: #fff;
}

.div_right th {
  color: #909399;
}

.td_tag {
  cursor: pointer;
}

#map2 {
  width: 100%;
  height: 100%;
  z-index: 1;
}

</style>

