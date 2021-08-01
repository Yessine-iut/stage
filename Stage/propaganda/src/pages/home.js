import Propaganda from "../images/propaganda.png";
import "../styles/home.css";
import Button from '../components/button';
import Card from '../components/card2';

//Quand on clique sur getStarted on se redirige vers le bas de la page
function redirection() {
  window.scrollBy(0,1000);
}


const Home = () => {
    return (
      <div >
          <div className="container2">
          <img className="bg" src={Propaganda}  alt="increase priority"style={{
            width:'100%',
                    }} />
                    <div className="text" >
        <h1>SIPS</h1>
        <h4>a System to Identify Propaganda Snippets </h4>
          <div>
      
           
         </div>
         <Button className="getStarted" id="btn2" onClick={()=>redirection()}>Get started</Button> 
                    </div> 

    </div>

<div className="boxes">
    <Card/>
    </div>
     
    </div>
    );
  };
  
  export default Home;