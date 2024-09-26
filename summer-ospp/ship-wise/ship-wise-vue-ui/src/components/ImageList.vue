<template>
  <div>
    <el-table ref="ImageList" :data="tableData1" height=375 highlight-current-row @row-dblclick="handleDoubleClick"
              :header-cell-style="tableHeaderStyle" :row-style="rowClass"
              @selection-change="handleSelectionChange" style="width: 100%;background-color: transparent;">
      <!-- <el-table-column type="index" min-width="150">
      </el-table-column> -->
      <el-table-column property="name" label="图像来源" min-width="80">
      </el-table-column>
      <el-table-column property="date" label="日期" click="resort" min-width="80" sortable>
      </el-table-column>
      <!-- <el-table-column property="time" label="时间" min-width="200">
      </el-table-column> -->
      <el-table-column property="latlng" label="经纬度" min-width="80">
      </el-table-column>
    </el-table>
    <div style="margin-top: 20px">
      <!-- <el-button @click="shipDetection()">载入所有小图</el-button>
      <el-button @click="returnMap()">返回地图</el-button> -->
      <!-- <el-button @click="loadMap()">显示检测结果</el-button> -->
      <!-- <el-button @click="resort()">日期升降序排序</el-button> -->
    </div>
    <div class="block">
      <!-- <span class="demonstration">方案1</span> -->
      <el-date-picker v-model="timeFrom" type="daterange" @change="setTime" :clearable="false" unlink-panels
                      value-format="yyyy-MM-dd HH:mm:ss" size="mini" range-separator="至" start-placeholder="开始日期"
                      end-placeholder="结束日期">
      </el-date-picker>
      <el-button type="primary" @click="search()" size="mini">搜索</el-button>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import {Aside} from 'element-ui';
import {getDetections, getSizes, getTifInfo, searchPoint} from "@/assets/js/api/common-api";

export default {
  data() {
    return {
      selectIndex: -1,
      tableData1: [],
      currentRow: null,
      latlng: [],
      score: [],
      file_path: [],
      ship_size: [], //船只尺寸
      ship_number: 1,
      json_number: 0,
      label: true, //用来标记升序和降序
      timeFrom: [],//时间数组，包含开始时间和结束时间
      queryInfo: {
        createTimeFrom: '',   // 开始时间
        createTimeTo: ''  //结束时间
      }
    }
  },

  methods: {
    tableHeaderStyle({row, column, rowIndex, columnIndex}) {
      if (rowIndex === 0) {
        return `
        background-color: rgba(85, 85, 255, 0.5);
        color: #fff;
        `;
      }
    },
    rowClass(row) {
      if ((row.rowIndex + 1) % 2 == 0) {
        return {background: "rgba(85, 170, 255, 0.5)", color: "#000000"};
      } else {
        return {background: "rgba(85, 170, 255, 0.5)", color: "#000000"};
      }
    },
    handleSelectionChange(val) {
      this.selectIndex = val;
    },
    async handleDoubleClick(row, column, event) {
      let json_number = 0
      let latIng = []
      let score = []
      let file_path = []
      let ship_size = []
      let b_box = []
      let ais_data = []
      let mmsi1 = []
      let ais_lating = []
      let ais_time = []
      this.$message('加载地图' + row.name + "中的船只信息");

      // AIS数据
      let {data: aisData} = await searchPoint(row.label, null)
      for (let item of aisData) {
        mmsi1.push(item[1])
        ais_time.push(item[0])
        ais_lating.push({
          lat: item[3],
          lng: item[2],
        })
      }
      let {data: shipClass} = await getSizes(row.label)
      let {data: tifInfo} = await getTifInfo(row.label)
      let {data: detectionData} = await getDetections(null, null, row.label)
      for (let item of detectionData) {
        if (row.label === item[0]) {
          let newList = {
            lat: item[2],
            lng: item[1],
          }
          latIng.push(newList); //经纬度
          score.push(item[4]);  //置信度
          file_path.push(item[3]) //子图路径
          ship_size.push(item[5])  //船只大小
          let bbox = {
            top: item[6],
            left: item[7],
            bottom: item[8],
            right: item[9]
          }
          b_box.push(bbox)
        }
      }
      this.$bus.emit('ship_data', latIng, file_path, tifInfo, b_box, ais_lating, mmsi1, ais_time, row.label)
      this.$bus.emit('change', latIng, score, ship_size)
      this.$bus.emit('ship_class', shipClass, row.latlng, row.date)
    },

    shipDetection() {
      //测试json数据
      this.$emit('test', 1)
      console.log(this.selectIndex);
    },

    returnMap() {

    },

    loadMap() {
      // const files = require.context("../public/picture", true, /\.jpg$/).keys();
      // console.log('file', files)
      this.$emit('picture_list', 1)
    },

    //接收所有图片的name并传入页面
    update1(data) {
      const tableData1 = []
      for (let item of data) {
        let name = item.split('_')[0]
        let date = item.split('_')[1]
        let latlng = item.split('_')[5] + item.split('_')[6]
        tableData1.push({name: name, date: date, latlng: latlng, label: item})
      }
      this.tableData1 = tableData1.sort((a, b) => b.date - a.date)
    },

    // resort() {
    //   if (this.label == true) {
    //     this.tableData1 = this.tableData1.sort((a, b) => a.date - b.date)
    //     this.label = false
    //   } else {
    //     this.tableData1 = this.tableData1.sort((a, b) => b.date - a.date)
    //     this.label = true
    //   }
    // }
    //按日期查询符合条件的大图
    search(queryInfo) {
      this.$emit('picture_list1', this.queryInfo)
      console.log("queryInfo", this.queryInfo)
    },
    setTime(e) {
      this.queryInfo.createTimeFrom = e[0]
      this.queryInfo.createTimeTo = e[1]
    }
  }
}
</script>
<style scoped>
.el-button--primary {
  background-color: #105EED !important;
  color: white !important;
  font-size: 18px;
  height: 44px;
}

::v-deep .el-table__empty-text {
  color: #ffffff;
}
</style>