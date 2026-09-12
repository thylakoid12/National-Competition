
import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const payloadPath=process.argv[2];
if(!payloadPath) throw new Error("请传入核验记录/tables.json 的路径");
const data=JSON.parse(await fs.readFile(payloadPath,"utf8"));
const output=data.output;
const audit=path.join(output,"核验记录");
await fs.mkdir(audit,{recursive:true});
const book=await SpreadsheetFile.importXlsx(await FileBlob.load(data.template));
const sourceCheck=await book.inspect({kind:"workbook,sheet,table",maxChars:1800,tableMaxRows:2,tableMaxCols:4});
await fs.writeFile(path.join(audit,"模板结构.ndjson"),sourceCheck.ndjson);
if(process.argv.includes("--render")){
  for(const name of ["计划购电量","充放电量","紧急购电量"]){
    const preview=await book.render({sheetName:name,range:name==="紧急购电量"?"A1:C5":"A1:F5",scale:1.2,format:"png"});
    await fs.writeFile(path.join(audit,"模板_"+name+".png"),new Uint8Array(await preview.arrayBuffer()));
  }
}

for(const [name,key] of [["计划购电量","purchase"],["充放电量","storage"],["紧急购电量","emergency"]]){
  const sheet=book.worksheets.getItem(name), rows=data[key];
  sheet.getUsedRange().clear({applyTo:"contents"});
  const area=sheet.getRangeByIndexes(0,0,rows.length,rows[0].length);
  area.values=rows;
  area.format.rowHeight=22;
  area.format.verticalAlignment="center";
  sheet.getRangeByIndexes(0,0,rows.length,1).format.columnWidth=15;
  sheet.getRangeByIndexes(1,0,rows.length-1,1).setNumberFormat("yyyy-mm-dd");
  sheet.getRangeByIndexes(0,0,rows.length,1).format.horizontalAlignment="center";
  if(key==="purchase"){
    sheet.getRangeByIndexes(0,1,rows.length,146).format.columnWidth=17;
    sheet.getRangeByIndexes(1,1,rows.length-1,146).setNumberFormat("0.0000");
  }else if(key==="storage"){
    sheet.getRangeByIndexes(0,1,rows.length,5).format.columnWidth=20;
    sheet.getRangeByIndexes(1,2,rows.length-1,2).setNumberFormat("0.0000");
    sheet.getRangeByIndexes(1,5,rows.length-1,1).setNumberFormat("0.0000");
    sheet.getRangeByIndexes(1,1,rows.length-1,1).format.horizontalAlignment="center";
    sheet.getRangeByIndexes(1,4,rows.length-1,1).format.horizontalAlignment="center";
  }else{
    sheet.getRangeByIndexes(0,1,rows.length,1).format.columnWidth=23;
    sheet.getRangeByIndexes(0,2,rows.length,1).format.columnWidth=20;
    sheet.getRangeByIndexes(1,1,rows.length-1,1).format.horizontalAlignment="center";
    sheet.getRangeByIndexes(1,2,rows.length-1,1).setNumberFormat("0.0000");
  }
  sheet.freezePanes.freezeRows(1);
}

const paper=Workbook.create();
for(const [name,rows] of Object.entries(data.paper)){
  const sheet=paper.worksheets.add(name);
  const area=sheet.getRangeByIndexes(0,0,rows.length,rows[0].length);
  area.values=rows;
  area.format.rowHeight=25;
  area.format.columnWidth=22;
  area.format.verticalAlignment="center";
  area.setNumberFormat("0.0000");
  sheet.showGridLines=false;
  function heading(row){
    const range=sheet.getRangeByIndexes(row,0,1,rows[0].length);
    range.format.fill="#E7EEF4";
    range.format.font.bold=true;
    range.format.rowHeight=29;
  }
  if(name==="评价期汇总"){
    sheet.getRangeByIndexes(0,0,rows.length,1).format.columnWidth=32;
    sheet.getRangeByIndexes(0,1,rows.length,1).format.columnWidth=53;
    sheet.getRange("B2:B4").format.horizontalAlignment="left";
    sheet.getRange("B3:B4").format.wrapText=true;
    sheet.getRange("A3:B4").format.rowHeight=46;
    heading(0);heading(4);
    for(let i=5;i<rows.length;i++){
      if(rows[i][0]==="光伏消纳率")sheet.getCell(i,1).setNumberFormat("0.0000%");
      if(rows[i][0].includes("天数")||rows[i][0].includes("时段数"))sheet.getCell(i,1).setNumberFormat("0");
      if(typeof rows[i][1]==="string"){
        sheet.getCell(i,1).format.wrapText=true;
        sheet.getRangeByIndexes(i,0,1,2).format.rowHeight=44;
      }
    }
  }else if(name==="表3_紧急购电"){
    heading(0);
    for(let c=0;c<8;c+=2){const col=sheet.getRangeByIndexes(0,c,rows.length,1);col.format.columnWidth=27;col.format.horizontalAlignment="center";}
    for(let c=1;c<8;c+=2)sheet.getRangeByIndexes(0,c,rows.length,1).format.columnWidth=18;
  }else{
    const timeColumns=name==="表1_购电结果"?[0,2,4]:[0,3];
    for(const c of timeColumns)sheet.getRangeByIndexes(0,c,rows.length,1).format.horizontalAlignment="center";
    const block=name==="表1_购电结果"?6:7;
    for(let row=0;row<rows.length;row+=block){
      heading(row);heading(row+1);
      sheet.getRangeByIndexes(row+1,0,1,6).format.horizontalAlignment="center";
    }
    if(name==="表1_购电结果")sheet.getRangeByIndexes(0,2,rows.length,1).format.columnWidth=27;
  }
}
await book.recalculate();
await paper.recalculate();
for(const [wb,name] of [[book,"result2"],[paper,"论文结果表"]]){
  const errors=await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:30},maxChars:2500});
  await fs.writeFile(path.join(audit,name+"_公式错误检查.ndjson"),errors.ndjson);
  const checks=await wb.inspect({kind:"table",range:name==="result2"?"计划购电量!A1:F5":"评价期汇总!A1:B8",include:"values,formulas",tableMaxRows:8,tableMaxCols:6,maxChars:2500});
  await fs.writeFile(path.join(audit,name+"_数值检查.ndjson"),checks.ndjson);
}
for(const [wb,name] of [[book,"result2.xlsx"],[paper,"论文结果表.xlsx"]]){
  const file=await SpreadsheetFile.exportXlsx(wb);
  await file.save(path.join(output,name));
}
const views=[
 [book,"计划购电量","A1:F6"],[book,"充放电量","A1:F13"],[book,"紧急购电量","A1:C12"],
 [paper,"评价期汇总","A1:B"+data.paper["评价期汇总"].length],[paper,"表1_购电结果","A1:F12"],
 [paper,"表2_储能结果","A1:F14"],[paper,"表3_紧急购电","A1:H"+data.paper["表3_紧急购电"].length]
];
const previews=process.argv.includes("--render")?views:process.argv.includes("--render-paper")?views.filter(v=>v[0]===paper&&v[1]!=="评价期汇总"):[];
for(const [wb,sheetName,range] of previews){
  const preview=await wb.render({sheetName,range,scale:1.5,format:"png"});
  await fs.writeFile(path.join(audit,sheetName+".png"),new Uint8Array(await preview.arrayBuffer()));
}
for(const name of ["result2.xlsx","论文结果表.xlsx"]){
  const sidecar=path.join(output,name+".inspect.ndjson");
  try{await fs.rename(sidecar,path.join(audit,name+".inspect.ndjson"));}
  catch(error){if(error.code!=="ENOENT")throw error;}
}
console.log("模型"+data.model+"：已导出全年提交表和论文Excel。");







