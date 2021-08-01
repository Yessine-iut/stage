import React from 'react';
//import {Reponse} from '../components/textInput'
//import data from "../json/jsonmockV4.json"
import {data} from '../components/textInput2'

/*import WordCloud from "@nhuson/react-d3-cloud";
import { Resizable } from "re-resizable";*/
import ReactWordcloud from "react-wordcloud";
import { saveAs } from "file-saver";
import Button from '../components/button';
import {Link} from "react-router-dom";

import Chkbox from '../components/chkbox2';
import "../styles/service2_1.css"
import "tippy.js/dist/tippy.css";
import "tippy.js/animations/scale.css";
import notFound from "../images/404.png"

export var word = [
 
];
const callbacks = {
  getWordColor: word => word.color,
  
}
const size = [600, 400];

const options = {
  colors: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
  enableTooltip: true,
  deterministic: false,
  fontFamily: "impact",
  fontSizes: [10, 25],
  fontStyle: "normal",
  fontWeight: "normal",
  padding: 1,
  rotations: 3,
  rotationAngles: [0, 10],
  scale: "sqrt",
  spiral: "archimedean",
  transitionDuration: 1000,
};

 class Service2_1 extends React.Component{
 render(){
  if(data===" "){
    return<><div style={{textAlign:'center',marginTop:'10%',fontSize:'60px'}}>The input text is empty.</div>
      <div style={{textAlign:'center',fontSize:'20px'}}>Please go back and put some text</div>
      <Link to="/PropagandaWordClouds" ><Button style={{marginLeft:"48%",marginBottom:"80px"}}>back</Button>
      </Link></>
  }
   if(data===undefined){
    return <> <img style={{
      display: 'block',margin:'auto',textAlign:'center',alignItems:'center',marginTop:'10%'
    }} src={notFound} alt="Error 404" /><Link to="/" ><Button style={{marginLeft:"48%",marginBottom:"80px"}}>back</Button>
    </Link></>
   }
  return (
    <div id="wc">
     {Datacloud()}
     
     <div style={{ width: "100%", height: "100%" }}>
        <ReactWordcloud  words={word}       callbacks={callbacks}
        options={options}
        size={size}/>
        
      </div> 
     
 <div id="resp" style={{
      
      position: 'absolute',
      left: '70%', 
      top:'95px',
      maxHeight:'450px',
      minHeight:'450px',
      overflowY:"scroll",
      backgroundColor:'rgb(217, 217, 217)'
      
    }}>
      {/*<Multiplechkbox/>*/}
      <table>
              <tbody>
      {creerChckBox()}
      </tbody>
      </table>
      
</div>
<div style={{textAlign:'center',marginBottom:"50px"}}>
<Button  onClick={()=>saveDynamicDataToFile()}>Export as a JSON file</Button>

<Link to="/" ><Button >back</Button>
</Link>
</div>
</div>  );
 }
}
 function saveDynamicDataToFile() {

  //var userInput = document.getElementById("myText").value;

  var blob = new Blob([JSON.stringify(data)], { type: "json/plain;charset=utf-8" });
  let today=new Date()
  let name="PropagandaWordClouds_"+today.getHours()+"h"+today.getMinutes()+"_"+today.getMonth()+"_"+today.getDate()+"_"+today.getFullYear()
  saveAs(blob, name);
}

function Datacloud(){
  word = [
 
  ];
  let spans=[]
  let name=""
  for(let i in data){
    name=i
  }
  for(let i in data[name]){

    for(let j in data[name][i]){
        if(j!=='text'){

        
      
   
      if(typeof(data[name][i][j].label)!=='object'){
       if(data[name][i][j].label!==0){
        let spanCourant={};
       spanCourant.text=data[name][i][j].span;
       spanCourant.value=data[name][i][j].probability;
       spanCourant.label=data[name][i][j].label;
        spans.push(spanCourant)
      }
      }else{
        for(let v=0;v<data[name][i][j].label.length;v++){

          let spanCourant={};
          spanCourant.text=data[name][i].text.substring(data[name][i][j].start_char[v],data[name][i][j].end_index[v]);
          spanCourant.value=data[name][i][j].probability[v];
          spanCourant.label=data[name][i][j].label[v];

          spans.push(spanCourant)
        
          
        }
      }
    }
  }

  }    
  let colors=["yellow","red","green","magenta","maroon","#7a7a7a","orange","lime","#9B5D9B","aqua","pink","#8486D4","#5A6309","#D1C1F0"]

  for(let i=0;i<spans.length;i++){
    if(spans[i].text.length>48){
      let mot=spans[i].text.substring(spans[i].text.lastIndexOf(" ")+1, spans[i].text.length);
      let motD=spans[i].text.substring(spans[i].text.substring(0, 48).lastIndexOf(" ")+1,48);
      word.push({text: spans[i].text.substring(0,spans[i].text.indexOf(motD))+"[...] "+mot,value: spans[i].value*100,color:colors[spans[i].label-1]}) //* 100

    }else
    word.push({text: spans[i].text,value: spans[i].value*100,color:colors[spans[i].label-1]}) //* 100

  }
  


        
}

function creerChckBox(){
  let propagandas=["Appeal_to_\nAuthority","Appeal_to\n_fear-prejudice","Bandwagon, \nReductio_ad_hitlerum","Black-\nand-White\n_Fallacy","Causal_\nOversimplification","Doubt","Exaggeration,\nMinimisation","Flag-\nWaving","Loaded_\nLanguage","Name_Calling,\nLabeling","Repetition","Slogans","Thought-\nterminating_\nCliches","Whataboutism\n,Straw_Men,\nRed_Herring"]
  let colors=["yellow","red","green","magenta","maroon","#7a7a7a","orange","lime","#9B5D9B","aqua","pink","#8486D4","#5A6309","#D1C1F0"]
  let chckBoxes=[]
  let tailleSpan=0;
  let spans=[]
  let labels=[]
  let sans_doublons
  let name=""
  for(let i in data){
    name=i
  }
  for(let i in data[name]){
      
      tailleSpan=Object.keys(data[name][i]).length

      for(let v=1;v<tailleSpan;v++){
        spans.push(data[name][i]["span_"+v])
  
      }
      spans.sort(function compare(a, b) {
        if (a.start_char < b.start_char )
           return -1;
        if (a.start_char > b.start_char )
           return 1;
        return 0;
      }); 
      for(let v=0;v<spans.length;v++){
          if(typeof(spans[v].label)!=='object'){
            labels.push(spans[v].label)

          }else{
            for(let p=0;p<spans[v].label.length;p++){
              labels.push(spans[v].label[p])

            }

          }
      }
      tailleSpan=0;
      sans_doublons = Array.from(new Set(labels));
  }
  var index = sans_doublons.indexOf(0);
  if (index > -1) {
    sans_doublons.splice(index, 1);
  }
  chckBoxes.push(<tr key={sans_doublons.length} ><td style={{
    display:"inline-flex",
    }}><mark id="propaganda99">Propaganda Techniques</mark></td></tr>)
    
  for(let i=0;i<sans_doublons.length;i++){
   chckBoxes.push(<Chkbox key={i} name={propagandas[sans_doublons[i]-1]} identifiant={'propaganda'+sans_doublons[i]} color={colors[sans_doublons[i]-1]} onClick={() => changerCloud('propaganda'+sans_doublons[i],this.color)}/>)
  }
 
  return chckBoxes;
  

}
export function changerCloud(name,color){
  for(let i=0;i<word.length;i++){

  
  if(document.querySelector('input[name="'+name+'"]').checked===true){
    if(color===word[i].color){
      let text=document.querySelectorAll("g text")
        for(let y=0;y<text.length;y++){
          if(text[y].getAttribute("fill")===color || color===RGBToHex(text[y].getAttribute("fill")).toUpperCase()){
            text[y].style.display='block'
          }
        }


  }
  }else{
      if(color===word[i].color){
          let text=document.querySelectorAll("g text")
            for(let y=0;y<text.length;y++){
              
              if(text[y].getAttribute("fill")===color || color===RGBToHex(text[y].getAttribute("fill")).toUpperCase()){
                text[y].style.display='none'
              }
            }


      }
  }
}



function RGBToHex(rgb) {
  let sep = rgb.indexOf(",") > -1 ? "," : " ";
  rgb = rgb.substr(4).split(")")[0].split(sep);

  let r = (+rgb[0]).toString(16),
      g = (+rgb[1]).toString(16),
      b = (+rgb[2]).toString(16);

  if (r.length === 1)
    r = "0" + r;
  if (g.length === 1)
    g = "0" + g;
  if (b.length === 1)
    b = "0" + b;

  return "#" + r + g + b;
}
}



 
  export default Service2_1;
