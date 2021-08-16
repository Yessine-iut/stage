import '../styles/card-style.css'
const Card=({imgsrc,title,link,nameButton,text})=>{
/**File card.jsx, These are the different cards created in the site (Home, Contact-us), used to define the styles of the maps and the architecture

*/
return(
<div className="card text-center shadow" id="shadow">
    <div className="overflow">
        <img src={imgsrc} height="180px" alt="contact_us" className="card-img-top"/>
    </div>
    <div className="card-body text-dark">
        <h4 className="card-title">{title}</h4>
        <p className="card-text text-secondary">
            {text}
        </p>

        
        <a href={link} className="btn btn-outline-success"> {nameButton}</a>
    </div>
</div>

    
);


}


export default Card;