import React, { useState } from 'react';

import InputText from '../components/textInput';
//import InputText from '../components/textInput3';
import { useHistory } from "react-router-dom"

import Loading from "../images/loading.gif"


const Service1 = () => {


    return (
      <>
      <div
      style={{
        display: 'flex',  justifyContent:'center', alignItems:'center',flexDirection:'column'
      }}>
        <div>
      <InputText history={useHistory} state={useState("")}>
      </InputText>
      </div>
      <div >
    
    </div>
    </div>
    <img  id="loading" style={{
      display: 'none',margin:'auto',textAlign:'center',alignItems:'center',marginTop:'15%'
    }} src={Loading} alt="loading..." />
    </>
    );
  };
  

  export default Service1;
