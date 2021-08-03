import React from 'react'
import "../styles/service1_1.css"
import "../styles/multipleChkbox.css"
import Chkbox from './chkbox';
import {data} from './textInput'

/**File multipleChkbox.js, used for service 1 to create the different chkboxes according to the data received */
class Multiplechkbox extends React.Component{
    render(){
        return(
        
        <div  style={{
            position: 'relative',
            left: '-15%',
            top:'95px',
            backgroundColor:'rgb(217, 217, 217)',
            minHeight:'450px',
            maxHeight:'450px',
            overflowY:"scroll"
                    }}> 
            
            <table>
                <tbody >
                 
        {creerChckBox()}
      
        </tbody>
        </table>
                
               
           

           

        </div>);
    }
   
 
    
    

}

function creerChckBox(){
  let propagandas=["Appeal_to_Authority","Appeal_to_fear-prejudice","Bandwagon,Reductio_ad_hitlerum","Black-and-White_Fallacy","Causal_Oversimplification","Doubt","Exaggeration,Minimisation","Flag-Waving","Loaded_Language","Name_Calling,Labeling","Repetition","Slogans","Thought-terminating_Cliches","Whataboutism,Straw_Men\n,Red_Herring"]
  let colors = {
    0: ["#ffffcc","#ffff99","#ffff66","#ffff32","#ffff00"],
    1: ["#ffe5e5","#ffb2b2","#ff7f7f","#ff6666","#ff0000"],
    2: ["#e5f2e5","#b2d8b2","#7fbf7f","#4ca64c","#008000"],
    3: ["#ffb2ff","#ff99ff","#ff7fff","#ff4cff","#ff00ff"],
    4:["#e5cccc","#D6ADAD","#cc9999","#b26666","#a34747"],
    5:["#e4e4e4","#c9c9c9","#afafaf","#949494","#7a7a7a"],
    6:["#ffedcc","#ffdb99","#ffc966","#ffb732","#ffa500"],
    7:["#ccffcc","#99ff99","#66ff66","#32ff32","#00ff00"],
    8:["#ebdeeb","#d7bed7","#c39dc3","#af7daf","#9b5d9b"],
    9:["#ccffff","#99ffff","#66ffff","#32ffff","#00ffff"],
    10:["#fff2f4","#ffe5ea","#ffd9df","#ffccd5","#ffc0cb"],
    11:["#e6e6f6","#cdceed","#b5b6e5","#9c9edc","#8486d4"],
    12:["#F3EEEA","#e6ded5","#d4c5b5","#c7b4a0","#c1ac95"],
    13:["#ece6f9","#ded3f4","#dacdf3","#d5c7f1","#d1c1f0"]};
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
      // We sort the different spans compared to their start
      spans.sort(function compare(a, b) {
        if (a.start_char < b.start_char )
           return -1;
        if (a.start_char > b.start_char )
           return 1;
        return 0;
      }); 
      for(let v=0;v<spans.length;v++){
          // If the span does not have several labels
          if(typeof(spans[v].label)!=='object'){
            labels.push(spans[v].label)

          }else{
            for(let p=0;p<spans[v].label.length;p++){
              labels.push(spans[v].label[p])

            }

          }
      }
      tailleSpan=0;
      // We remove the duplicates in the labels
      sans_doublons = Array.from(new Set(labels));
  }
  var index = sans_doublons.indexOf(0);
  // We remove everything that is equal to 0
  if (index > -1) {
    sans_doublons.splice(index, 1);
  }
   chckBoxes.push(<tr  key={15} ><td style={{
    display:"inline-flex",
    }}><mark id="propaganda99">Propaganda Techniques</mark></td></tr>)


  for(let i=0;i<sans_doublons.length;i++){
    // We create the checkboxes being careful not to create the checkboxes for the sentences that do not have a label
    if(sans_doublons[i]!==0){
   chckBoxes.push(<Chkbox key={i} name={propagandas[sans_doublons[i]-1]} identifiant={'propaganda'+sans_doublons[i]} color={colors[sans_doublons[i]-1]} label={""+sans_doublons[i]}/>)
  }

  }
  chckBoxes.push(<tr key={16} ><td style={{
    display:"inline-flex",
    }}><mark style={{border:"1px solid black",marginLeft:"14px",paddingLeft:"13px",paddingRight:"13px",marginRight:"5px",  marginBottom: '20px'}} id="propaganda99" >Bold</mark>Multiple Propaganda</td></tr>)
 
  return chckBoxes;
  

}

export default Multiplechkbox;
