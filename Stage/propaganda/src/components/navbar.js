import React from "react";
import "../styles/navbar.css";
import Image from "../images/i3s.png";
import Cnrs from "../images/cnrs.png";
import Univ from "../images/cotedazur.png";
import Wimmics from "../images/wimmics.png";
import Inria from "../images/inria.png";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import { Navbar, Nav, NavDropdown } from "react-bootstrap";

/**Fichier navbar.js, c'est la navbar du site web */
class BootstrapNavbar extends React.Component {
  render() {
    return (
      <div onLoad={() => handleResize()}>
        <div className="row"></div>
        <div
          className="col-md-12"
          style={{
            marginBottom: "-23px",
          }}
        >
          <Router>
            <Navbar
              style={{ backgroundColor: "#D9D9D9" }}
              /*bg="light"*/ variant="light"
              expand="lg"
              sticky="top"
            >
              <div className="logo">
                <Navbar.Brand href="https://univ-cotedazur.eu/">
                  <img src={Univ} alt="logo" height="60"></img>
                </Navbar.Brand>
                <Navbar.Brand href="http://www.cnrs.fr/">
                  <img src={Cnrs} alt="logo" height="60"></img>
                </Navbar.Brand>
                <Navbar.Brand href="https://www.inria.fr/fr">
                  <img src={Inria} alt="logo" height="60"></img>
                </Navbar.Brand>
                <Navbar.Brand href="https://www.i3s.unice.fr/">
                  <img src={Image} alt="logo" height="60"></img>
                </Navbar.Brand>

                <Navbar.Brand href="https://team.inria.fr/wimmics/">
                  <img src={Wimmics} alt="logo" height="60"></img>
                </Navbar.Brand>
              </div>

              <Navbar.Toggle aria-controls="basic-navbar-nav" />

              <Navbar.Collapse id="basic-navbar-nav">
                <div className="navbarY">
                  <Nav className="mr-auto">
                    <Nav.Link href="/" /**style={{color:'black'}}*/>
                      Home
                    </Nav.Link>
                    <NavDropdown title="Services" id="basic-nav-dropdown">
                      <NavDropdown.Item
                        href="/PropagandaSnippetsDetection" /**style={{color:'black'}}*/
                      >
                        Snippets Detection
                      </NavDropdown.Item>
                      <NavDropdown.Item
                        href="/PropagandaWordClouds" /**style={{color:'black'}}*/
                      >
                        Propaganda Word Cloud
                      </NavDropdown.Item>

                      {/*<NavDropdown.Divider />*/}
                      {/*<NavDropdown.Item href="#action/3.4">Separated link</NavDropdown.Item>*/}
                    </NavDropdown>

                    <Nav.Link href="/about" /**style={{color:'black'}}*/>
                      About
                    </Nav.Link>
                    <Nav.Link href="/contact-us" /**style={{color:'black'}}*/>
                      Contact Us
                    </Nav.Link>
                  </Nav>
                </div>
              </Navbar.Collapse>
              {/*<Navbar.Brand  className="logo2" style={{display:'visible',marginRight:"8rem"}} href="https://www.inria.fr/fr">  
                                   <img src={Inria} alt="logo" height="60"></img>
                                        </Navbar.Brand>*/}
            </Navbar>

            <br />

            <Switch>
              <Route exact path="/">
                {/*<Home />*/}
              </Route>

              <Route path="/about-us">{/* <AboutUs />*/}</Route>

              <Route path="/contact-us">{/*<ContactUs />*/}</Route>
            </Switch>
          </Router>

          {/*</div>*/}
        </div>

        {window.addEventListener("resize", handleResize)}
      </div>
    );
  }
}

function handleResize() {
  if (window.innerWidth < 992) {
    document.querySelector(".navbarY").style.position = "relative";
    document.querySelector(".navbarY").style.right = 0;
  } else {
    document.querySelector(".navbarY").style.position = "absolute";
    document.querySelector(".navbarY").style.right = 10;
  }
}
export default BootstrapNavbar;
