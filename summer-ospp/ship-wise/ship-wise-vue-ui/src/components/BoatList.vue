<template>
  <div style="overflow:auto">
    <el-table :data="tableData" style="width: 100%;background-color: transparent;"
              :header-cell-style="tableHeaderStyle" :row-style="rowClass" highlight-current-row @row-click="clickTr">
      <!-- <el-table-column type="index" label="编号" width="100%" >
      </el-table-column> -->
      <el-table-column prop="gis" label="经纬度" width="100%" sortable>
      </el-table-column>
      <el-table-column prop="accurate" label="可信度" sortable>
      </el-table-column>
      <el-table-column prop="ship_size" label="船只大小" sortable>
      </el-table-column>
    </el-table>
  </div>
</template>

<script>
export default {
  data() {
    return {
      tableData: [],
      label: true,
    }
  },
  created() {
    this.$bus.on('change', this.update)
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

    //点击跳转功能
    async clickTr(row, event, column) {
      const a = row["gis"].split(",")[0];
      const b = row["gis"].split(",")[1];
      console.log(a, b)
      this.$bus.emit('ship_data1', b, a)
      // this.map.panTo([b,a]);
      // alert(row["name"]);//获取各行id的值
    },

    update(latIng, score, shipsize) {
      let tableData = []
      // for (let lat of latIng) {
      //     tableData.push({ ID: item.船只子图id, gis: `${item.经度},${item.纬度}`, accurate: item.可信度 })
      //     // tableData = [...tableData, { ID: item.船只子图id, gis: `${item.经度},${item.纬度}`, accurate: item.可信度 }]
      // }
      for (let i = 0; i < score.length; i++) {
        tableData.push({gis: `${latIng[i].lat},${latIng[i].lng}`, accurate: score[i], ship_size: shipsize[i]})
        // tableData = [...tableData, { ID: item.船只子图id, gis: `${item.经度},${item.纬度}`, accurate: item.可信度 }]
      }
      console.log("tableData:", tableData)
      this.tableData = tableData
    },

    resort() {
      if (this.label == true) {
        this.tableData = this.tableData.sort((a, b) => a.date - b.date)
        this.label = false
      } else {
        this.tableData = this.tableData.sort((a, b) => b.date - a.date)
        this.label = true
      }
    }
  }
}

</script>
<style scoped>
::v-deep .el-table__empty-text {
  color: #ffffff;
}
</style>