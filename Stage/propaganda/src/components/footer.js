import React from 'react'
import './footer.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {  /*faFacebookF, faGithub, faLinkedin,*/ faTwitter , faYoutube } from '@fortawesome/free-brands-svg-icons';

/**File footer.js, it's the footer of each page of the site*/
function footer () {
  return (
    <div id="footer" className="site-footer">
    <div className="container">
      <div className="row">
        <div className="col-sm-12 col-md-6">
          <h6>About</h6>
          <p className="text-justify">Disinformation is spreading online through news, social media and etc. Based on NLP techniques, we explore such disinformation via rhetorical, psychological and strategies to detect such propaganda text.</p>
        </div>

        <div className="col-xs-6 col-md-3">
          <h6>Services</h6>
          <ul className="footer-links">
            <li><a href="./PropagandaSnippetsDetection" style={{color:"wheat"}}>Propaganda Snippets Detection</a></li>
            <li><a href="./PropagandaWordClouds" style={{color:"wheat"}}>Propaganda Word Cloud</a></li>
           
          </ul>
        </div>

        <div className="col-xs-6 col-md-3">
          <h6>Quick Links</h6>
          <ul className="footer-links">
            <li><a href="./about" style={{color:"wheat"}}>About</a></li>
            <li><a href="./contact-us" style={{color:"wheat"}}>Contact Us</a></li>
            
          </ul>
        </div>
      </div>
     
    </div>
    <div className="container">
      <div className="row">
        <div className="col-md-8 col-sm-6 col-xs-12">
          <p className="copyright-text">Copyright &copy; 2021 All Rights Reserved by 	&nbsp;
       <a href="https://www.i3s.unice.fr/" style={{color:"wheat"}}>I3s</a>
       <><br/>Designed by Scanfcode.</>

          </p>
        </div>

        <div className="col-md-4 col-sm-6 col-xs-12">
          <ul className="social-icons">
            <li><a className="twitter" href="https://twitter.com/wimmics?lang=en"><FontAwesomeIcon icon={faTwitter}/></a></li>
            <li><a className="youtube" href="https://www.youtube.com/channel/UCwo21i7ELnTGZ72kHzOq7kQ"><FontAwesomeIcon icon={faYoutube}/></a></li>
          </ul>
        </div>
      </div>
    </div>
</div>
  )
}
export default footer;