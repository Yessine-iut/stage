import React from 'react';
import {data} from '../components/textInput'
//import data from "../json/jsonmockV4.json"
import "../styles/service1_1.css"
import Button from '../components/button';
import Multiplechkbox from '../components/multipleChkbox';
import {Link} from "react-router-dom";
//import file from "./json.txt"
import { saveAs } from "file-saver";
import notFound from "../images/404.png"




/*const Service1_1 = () => {
    return (
    {render}

    );
  };*/
  class Service1_1 extends React.Component{
    render(){
      if(data===" "){
        return<><div style={{textAlign:'center',marginTop:'10%',fontSize:'60px'}}>The input text is empty.</div>
        <div style={{textAlign:'center',fontSize:'20px'}}>Please go back and put some text.</div>
        <Link to="/PropagandaSnippetsDetection" ><Button style={{marginLeft:"48%",marginBottom:"80px"}}>back</Button>
        </Link></>
      }
      if(data===undefined){
        return <> <img style={{
          display: 'block',margin:'auto',textAlign:'center',alignItems:'center',marginTop:'10%'
        }} src={notFound} alt="Error 404" /><Link to="/" ><Button style={{marginLeft:"48%",marginBottom:"80px"}}>back</Button>
        </Link></>
      }
      return (
        <div style={{height:"700px"}}
  >
          <div>
      
          
        <div id="res" style={{
        /*right:'15%',*/
        lineHeight:'25px',
       width:'76%',
      
      
        }}><div style={{marginTop:"70px",backgroundColor:"white",minHeight:"500px",/* marginTop:'-23px',*/position:"relative",maxHeight:"500px",overflowY:'scroll',paddingLeft: '8px'}}>{texte2()}</div>
        <div id="btn" className="button"
        style={{
          
          position: 'relative',
          left: '42%',
          top:'2em',
          padding:'4px 5px'
          
          
        }}>
           <Button style={{marginRight:"10px"}} onClick={()=>saveDynamicDataToFile()}>Export JSON file</Button>
  
  <Link to="/" ><Button >back</Button>
  </Link>
        </div>
       
  
        </div> 
        <div id="test" style={{
          
          position: 'absolute',
          left: '40%', 
          
        }}>
        
  </div>
        <div style={{
          
          position: 'absolute',
          left: '80%',      
          
        }}>
             <Multiplechkbox/>
           </div>
           { /* <mark>test</mark>*/}
        </div>
  
      </div>
  
      );
    }
    
  }
  
  
  /*function download(text, name, type) {
    var a = document.getElementById("a");
    var file = new Blob([text], {type: type});
    a.href = URL.createObjectURL(file);
    a.download = name;
  }*/
function texte2(){
  let propagandas=["Appeal_to_Authority","Appeal_to_fear-prejudice","Bandwagon,Reductio_ad_hitlerum","Black-and-White_Fallacy","Causal_Oversimplification","Doubt","Exaggeration,Minimisation","Flag-Waving","Loaded_Language","Name_Calling,Labeling","Repetition","Slogans","Thought-terminating_Cliches","Whataboutism,Straw_Men,Red_Herring"]
  let spans=[]
  let res=[]
 // let taillePropaganda=0;
  let tailleSpan=0;
  let text=""
  let name=""
  for(let i in data){
    name=i
  }
  for(let i in data[name]){
 // }
  //Pour chaque propagande
  //for(let i=1;i<taillePropaganda+1;i++){
    tailleSpan=Object.keys(data[name][i]).length
    text=data[name][i].text
    for(let v=1;v<tailleSpan;v++){
      spans.push(data[name][i]["span_"+v])

    }
    spans.sort(function compare(a, b) {
      if(typeof(a.start_char)!=='object'&& typeof(b.start_char)!=='object'){

     
      if (a.start_char < b.start_char )
         return -1;
      if (a.start_char > b.start_char )
         return 1;
      return 0;
    }
    else if (typeof(a.start_char)==='object' && typeof(b.start_char)!=='object'){
      if (a.start_char[0] < b.start_char )
      return -1;
   if (a.start_char[0] > b.start_char )
      return 1;
   return 0;
    }
    else if(typeof(a.start_char)!=='object' && typeof(b.start_char)==='object'){
      if (a.start_char < b.start_char[0] )
      return -1;
   if (a.start_char> b.start_char[0] )
      return 1;
   return 0;
    }
    else if(typeof(a.start_char)==='object' && typeof(b.start_char)==='object'){
      if (a.start_char[0] < b.start_char[0] )
      return -1;
   if (a.start_char[0]> b.start_char[0] )
      return 1;
   return 0;
    }
    else return 0
    }); 
    let curseur=0;
    for(let v=0;v<spans.length;v++){
      let spanCourant=spans[v]
      let spanText=spans[v].span
     
      if(tailleSpan-1===1 && typeof(spanCourant.label)!=='object'){

        res.push(text.substring(0,spanCourant.start_char))
        let id="propaganda"+spans[v].label
        let label1=[]
        label1.push(spans[v].label)
        let couleuri=intensite(spans[v].label,spans[v].probability)
        let title=propagandas[spans[v].label-1]+" - Probability: "+spans[v].probability
        if(couleuri==='none'){
          res.push(<mark style={{
            backgroundColor:couleuri,
          }} probability={spans[v].probability} key={v+id+label1+title+spanCourant.span+curseur} id={id} labels={label1}>{spanCourant.span} </mark>)
        }else
        res.push(<mark style={{backgroundColor:couleuri}} key={v+id+label1+title+spanText+curseur} id={id} couleur={couleuri} labels={label1} title={title} title2={title} probability={spans[v].probability}>{spanText}</mark>)
        res.push(text.substring(spanCourant.end_index,text.length)+" ")

      }else{
        //On regarde si ça se chevauche
        let over=0;
        

          if(typeof(spanCourant.label)==='object'){
            let SpansOverlap=[]
            let p=v;
            while(p<tailleSpan-1){
              if(spans[p].probability!==0){
              SpansOverlap.push(spans[p])
            }

              p++;
            }
             /*SpansOverlap.push(spanCourant)
             SpansOverlap.push(spanSuivant)*/
             

          intervalles(SpansOverlap,text,res,curseur);


            over=1;
            v=tailleSpan
          }
         
          

        
        if(over!==1){
          over=0;
          let id="propaganda"+spanCourant.label
          let label1=[]
          label1.push(spanCourant.label)
          let title=propagandas[spans[v].label-1]+" - Probability: "+spans[v].probability

          res.push(text.substring(curseur,spanCourant.start_char))
          let couleuri=intensite(spans[v].label,spans[v].probability)
          if(couleuri==='none'){
            res.push(<mark style={{
              backgroundColor:couleuri,
            }} probability={spans[v].probability} key={v+id+label1+title+spanCourant.span+curseur} id={id} labels={label1}>{spanCourant.span} </mark>)
          }else
          res.push(<mark style={{
            backgroundColor:couleuri,
          }} probability={spans[v].probability} key={v+id+label1+title+spanCourant.span+curseur} id={id} labels={label1} title={title} title2={title}>{spanCourant.span} </mark>)
          curseur=spanCourant.end_index;
          // A enlever si c'est faux
          if(spanCourant.end_index<text.length-1 && v+1>spans.length-1){
            res.push(text.substring(spanCourant.end_index,text.length))

          }


        }
      }
    }
    
    curseur=0;
    tailleSpan=0;
    spans.splice(0,spans.length)
  }
  return res;
}

  function intensite(label,probability){
    if(label===0){
      return "none"
    }
    var obj1 = {
    1: ["#ffffcc","#ffff99","#ffff66","#ffff32","#ffff00"],
    2: ["#ffe5e5","#ffb2b2","#ff7f7f","#ff6666","#ff0000"],
    3: ["#e5f2e5","#b2d8b2","#7fbf7f","#4ca64c","#393"],
    4: ["#ffb2ff","#ff99ff","#ff7fff","#ff4cff","#ff00ff"],
    5:["#e5cccc","#D6ADAD","#cc9999","#b26666","#ad3333"],
    6:["#e4e4e4","#c9c9c9","#afafaf","#949494","#7a7a7a"],
    7:["#ffedcc","#ffdb99","#ffc966","#ffb732","#ffa500"],
    8:["#ccffcc","#99ff99","#66ff66","#32ff32","#00ff00"],
    9:["#ebdeeb","#d7bed7","#c39dc3","#af7daf","#9b5d9b"],
    10:["#ccffff","#99ffff","#66ffff","#32ffff","#00ffff"],
    11:["#fff2f4","#ffe5ea","#ffd9df","#ffccd5","#ffc0cb"],
    12:["#e6e6f6","#cdceed","#b5b6e5","#9c9edc","#8486d4"],
    //13:["#dedfcd","#bdc09c","#9ca16b","#7a823a","#5a6309"],
    13:["#F3EEEA","#e6ded5","#d4c5b5","#c7b4a0","#c1ac95"],

    14:["#ece6f9","#ded3f4","#dacdf3","#d5c7f1","#d1c1f0"]};
    
    probability=Math.round(probability*100/25)
    return(obj1[label][probability])
    /*for(let i=0;i<15;i++){
      let name='propaganda'+i;
      let span = document.querySelectorAll("#res #"+name);
      span.forEach(function(sp) {
        /*sp.style.backgroundColor=color;
        sp.title=sp.getAttribute("title2")*/
 /*});
    }*/
   

  }
  function saveDynamicDataToFile() {

    //var userInput = document.getElementById("myText").value;

    var blob = new Blob([JSON.stringify(data)], { type: "json/plain;charset=utf-8" });
    let today=new Date()
    let name="PropagandaSnippetsDetection_"+today.getHours()+"h"+today.getMinutes()+"_"+today.getMonth()+"_"+today.getDate()+"_"+today.getFullYear()
    saveAs(blob, name);
}

  
  function intervalles(spans,texte,res,curseur){
    let propagandas=["Appeal_to_Authority","Appeal_to_fear-prejudice","Bandwagon,Reductio_ad_hitlerum","Black-and-White_Fallacy","Causal_Oversimplification","Doubt","Exaggeration,Minimisation","Flag-Waving","Loaded_Language","Name_Calling,Labeling","Repetition","Slogans","Thought-terminating_Cliches","Whataboutism,Straw_Men,Red_Herring"]

   let inter=[]

   for(let j=0;j<spans.length;j++){
    
     if(typeof(spans[j].label)==="object"){
    for(let i=0;i<spans[j].label.length;i++){
      inter.push(spans[j].start_char[i])
      inter.push(spans[j].end_index[i])
    }
  }else{
    inter.push(spans[j].start_char)
    inter.push(spans[j].end_index)
  }
   }
  
   const byValue = (a,b) => a - b;
   const sorted=[...inter].sort(byValue);
   res.push(texte.substring(curseur,sorted[0]))
   curseur=sorted[0];
   let labels1=[]
   
   for(let i=1;i<sorted.length;i++){
    labels1=labels(spans,curseur+1);
        if(labels1.length===1){
      let id="propaganda"+labels1[0];
      if(curseur!==sorted[i]){
        let label=labels(spans,curseur+1);
        let probability=probabilities(spans,curseur+1)
       let title=propagandas[label-1]+" - Probability: "+probability
       let couleuri=intensite(label,probability)

        res.push(<mark probability={probability} style={{
          backgroundColor:couleuri,
        }} key={id+labels(spans,curseur+1)+probability+title+curseur} title={title} title2={title} id={id} labels={label} labels2={label}>{texte.substring(curseur,sorted[i])}</mark>)

      }
    }else if(labels1.length===0){
      res.push(texte.substring(curseur,sorted[i]))

    }
    else{
      
      let id="propaganda"+99;
      if(curseur!==sorted[i]){
        let label=labels(spans,curseur+1);
        let probability=probabilities(spans,curseur+1)
        let title="";
        for(let f=0;f<label.length;f++){
          if(f!==label.length-1){

          title=title+propagandas[label[f]-1]+" - Probability: "+probability[f]+"\n"
        }
        else
        title=title+propagandas[label[f]-1]+" - Probability: "+probability[f]+"\n"

        }
       /* data-for="main"
        data-tip="<p>test</p> <p>lol</p>" data-html="true" data-multiline="true"
        data-type="info" */
        res.push(<mark probability={probability} key={id+labels(spans,curseur+1)+probabilities+title+curseur} id={id} labels={label} labels2={label} title={title} title2={title}>{texte.substring(curseur,sorted[i])}</mark>)
        

      }
      

    }
    curseur=sorted[i];
    labels1.splice(0,labels1.length)

   }
   labels1=labels(spans,curseur+1);
   if(labels1.length===0){
     res.push(texte.substring(curseur,texte.length))
   }else if(labels1.length===1){
    let id="propaganda"+labels1[0];
    let label=labels(spans,curseur+1);
    let probability=probabilities(spans,curseur+1)
    let title="";

    title=title+propagandas[label-1]+" - Probability: "+probability+"\n"
    let couleuri=intensite(label,probability)
      if(couleuri==='none'){
        res.push(<mark style={{
          backgroundColor:couleuri,
        }} probability={probability} key={id+labels(spans,curseur+1)} id={id} labels={labels(spans,curseur+1)} labels2={labels(spans,curseur+1)}>{texte.substring(curseur,texte.length)}</mark>)          }else
    res.push(<mark style={{
      backgroundColor:couleuri,
    }} probability={probability} key={id+labels(spans,curseur+1)} id={id} labels={labels(spans,curseur+1)} labels2={labels(spans,curseur+1)} title={title} title2={title}>{texte.substring(curseur,texte.length)}</mark>)
   }else{
    let id="propaganda"+99;
    let label=labels(spans,curseur+1);
    let probability=probabilities(spans,curseur+1)
    let title="";
    for(let f=0;f<label.length;f++){
      title=title+propagandas[label[f]-1]+" - Probability: "+probability[f]+"\n"

    }

    res.push(<mark probability={probability} key={id+labels(spans,curseur+1)} id={id} labels={labels(spans,curseur+1)} labels2={labels(spans,curseur+1)} title={title} title2={title}>{texte.substring(curseur,texte.length)}</mark>)
   }
  }

 function labels(spans,curseur){
    let labels=[]
    for(let t=0;t<spans.length;t++){
    if(typeof(spans[t].label)==='object'){

    
    for(let i=0;i<spans[t].label.length;i++){
      if(curseur>=spans[t].start_char[i] && curseur<=spans[t].end_index[i]){
        labels.push(spans[t].label[i])
      }
    }
  }
  else{
    if(curseur>=spans[t].start_char && curseur<=spans[t].end_index){
    labels.push(spans[t].label)
    }
  }
}
return labels
  }
 function probabilities(spans,curseur){
    let labels=[]
    for(let t=0;t<spans.length;t++){
    
    if(typeof(spans[t].label)==='object'){

    
    for(let i=0;i<spans[t].label.length;i++){
      if(curseur>=spans[t].start_char[i] && curseur<=spans[t].end_index[i]){
        labels.push(spans[t].probability[i])
      }
    }
  }
  else{
    if(curseur>=spans[t].start_char && curseur<=spans[t].end_index){
    labels.push(spans[t].probability)
    }
  }
}
return labels
  }


  export default Service1_1;
