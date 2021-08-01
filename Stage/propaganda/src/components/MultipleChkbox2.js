import React from 'react'
import "../styles/service1_1.css"
import "../styles/multipleChkbox.css"
import Chkbox from './chkbox2';
//import data from "../json/jsonmockV3.json"
import {data} from '../components/textInput'

class Multiplechkbox extends React.Component{
    render(){
        return(
        
        <div  style={{
            position: 'relative',
            left: '20%',
                    }}> 
            
           {/* <table>
                <tbody>
        {creerChckBox()}
        </tbody>
        </table>*/}
                
               
           

           

        </div>);
    }
   
 
    
    

}
function creerChckBox(){
    let propagandas=["Appeal_to_Authority","Appeal_to_fear-prejudice","Bandwagon,Reductio_ad_hitlerum","Black-and-White_Fallacy","Causal_Oversimplification","Doubt","Exaggeration,Minimisation","Flag-Waving","Loaded_Language","Name_Calling,Labeling","Repetition","Slogans","Thought-terminating_Cliches","Whataboutism,Straw_Men,Red_Herring"]
    let colors=["yellow","red","green","magenta","maroon","sienna","orange","lime","#9B5D9B","aqua","pink","#8486D4","#5A6309","#D1C1F0"]
    let chckBoxes=[]
    //let taillePropaganda=0;
    let tailleSpan=0;
    let spans=[]
    let labels=[]
    let sans_doublons
    for(let i in data.article701225819){
        //taillePropaganda++;
     // }
   // for(let i=1;i<taillePropaganda+1;i++){
    tailleSpan=Object.keys(data.article701225819[i]).length

        for(let v=1;v<tailleSpan;v++){
          spans.push(data.article701225819[i]["span_"+v])
    
        }
        spans.sort(function compare(a, b) {
          if (a.start_char < b.start_char )
             return -1;
          if (a.start_char > b.start_char )
             return 1;
          return 0;
        }); 
        for(let v=0;v<spans.length;v++){
            labels.push(spans[v].label)
        }
        tailleSpan=0;
        sans_doublons = Array.from(new Set(labels));
    }
    var index = sans_doublons.indexOf(0);
    if (index > -1) {
      sans_doublons.splice(index, 1);
    }
    for(let i=0;i<sans_doublons.length;i++){
     chckBoxes.push(<Chkbox key={i} name={propagandas[sans_doublons[i]-1]} identifiant={'propaganda'+sans_doublons[i]} color={colors[sans_doublons[i]-1]}/>)
    }
   /* chckBoxes.push(<Chkbox name="propaganda 1" identifiant='propaganda1' />)
    chckBoxes.push(<Chkbox name="propaganda 2"  identifiant='propaganda2' />)
    chckBoxes.push(<Chkbox name="propaganda 3" identifiant='propaganda3' />)*/
    return chckBoxes;
    
 
}
export default Multiplechkbox;