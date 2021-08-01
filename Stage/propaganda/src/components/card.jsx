import './card-style.css'
const Card=({imgsrc,title,link,nameButton,text})=>{
/**Fichier card.jsx, Ce sont les différentes cartes créees dans le site (Home, Contact-us), permet de définir les styles des cartes et l'architecture*/
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