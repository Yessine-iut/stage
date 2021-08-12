import './App.css';
import { BrowserRouter as Router, Switch, Route } from 'react-router-dom';
import Home from './pages/home.js';
import NavBar from './components/navbar.js';
import Service1 from './pages/service1';
import Service1_1 from './pages/service1_1';
import Service2 from './pages/service2';
import Service2_2 from './pages/service2_1';
import ContactUs from './pages/contact-us';
import About from './pages/about';
import Footer from './components/footer.js';













function App() {
  
  return (
    <> 
    <NavBar/>
     <Router>
      <Switch>
        <Route path='/' exact component={Home} />
        <Route path='/PropagandaTechniquesClassification' component={Service1} />
        <Route path='/PropagandaTechniquesClassification2' component={Service1_1} />
        <Route path='/PropagandaWordClouds' component={Service2} />
        <Route path='/PropagandaWordClouds2' component={Service2_2} />
        <Route path="/about" component={About}/>
        <Route path='/contact-us' component={ContactUs} />



       

      </Switch>
    
      

    </Router>
    <Footer></Footer>



    </>
  
  )
}


export default App;
