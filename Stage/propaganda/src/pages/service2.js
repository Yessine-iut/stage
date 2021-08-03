import React from 'react';
import InputText from '../components/textInput2';
import Loading from "../images/loading.gif"


//File service2.js, the first page of the service 2 of the website
const Service2 = () => {
    return (
      <>
      <div
       style={{
        display: 'flex',  justifyContent:'center', alignItems:'center',flexDirection:'column'
      }}>
        <div>
      <InputText value="">
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
  

  export default Service2;
