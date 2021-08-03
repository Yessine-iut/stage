import Card from "../components/card";
import img1 from "../images/Elena.jpeg";
import img2 from "../images/Serena.jpg";
import img3 from "../images/Vorakit.jpg";

import img4 from "../images/Yessine.jpg";
import "../styles/contact-us.css";
// The Contact-Us page of the website
const ContactUs = () => {
  return (
    <>
      <div id="h1" style={{ marginBottom: "40px", fontFamily: "Georgia" }}>
        <div
          style={{
            backgroundColor: "rgba(0, 0, 0, 0.8)",
            width: "70%",
            margin: "0 auto",
          }}
        >
          <h1
            className="snippets"
            style={{ fontFamily: "Georgia" }}
          >
            Who we are
          </h1>

          <h2
            className="test"
            style={{
              textAlign: "center",
              fontSize: "20px",
              marginRight: "10%",
              marginLeft: "10%",
              textShadow: "none",
              paddingBottom: "20px",
            }}
          >
            We are a group of researchers and students at Université Côte
            d'Azur, INRIA Sophia Antipolis, CNRS, I3S laboratory. We simple have
            a passion to make AI and language accessible to everyday life.
          </h2>
        </div>
      </div>

      <div
        style={{ marginTop: "-70px" }}
        className="container-fluid d-flex justify-content-center"
      >
        <div className="row" id="cardResponsive">
          <div className="col-md-3">
            <Card
              imgsrc={img1}
              title="Elena Cabrio"
              link="https://www-sop.inria.fr/members/Elena.Cabrio/"
              nameButton="More"
              text="Université Côte d’Azur, CNRS, Inria, I3S, France"
            />
          </div>
          <div className="col-md-3">
            <Card
              imgsrc={img2}
              title="Serena Villata"
              link="https://www.i3s.unice.fr/~villata/index.html"
              nameButton="More"
              text="Université Côte d’Azur, CNRS, Inria, I3S, France"
            />
          </div>
          <div className="col-md-3">
            <Card
              imgsrc={img3}
              title="Vorakit Vorakitphan"
              nameButton="More"
              link="https://sites.google.com/view/vorakitvorakitphan"
              text="Université Côte d’Azur, CNRS, Inria, I3S, France"
            />
          </div>
          <div className="col-md-3">
            <Card
              imgsrc={img4}
              title="Yessine Ben El Bey"
              link="https://www.linkedin.com/in/yessinebenelbey/"
              nameButton="More"
              text="Université Côte d'Azur"
            />
          </div>
        </div>
      </div>
    </>
  );
};

export default ContactUs;
