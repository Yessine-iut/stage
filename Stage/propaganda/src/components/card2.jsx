import React,{Component} from 'react';
import Card from './card';
import img2 from '../images/wordcloud.jpg'
import img3 from '../images/text_high.png'



/**Fichier card2.jsx, Class Cards used to create a Card and its content.*/
class Cards extends Component{
    render(){
        return(
<div className="container-fluid d-flex justify-content-center">
    <div className="row">
        <div className="col-md-6">
        <Card imgsrc={img3} title="Propaganda Snippets Detection" link="./PropagandaSnippetsDetection"  nameButton="Run" text="Input a free text to identify input free text to identify and classify propaganda techniques."/>
        
        </div>
        <div className="col-md-6">
<Card imgsrc={img2} title="Propaganda Word Clouds" link="./PropagandaWordClouds" nameButton="Run" text="Display propaganda snippets as word clouds from a free text."/>
        </div>
       
    </div>
</div>
        );
    }
}
export default Cards;