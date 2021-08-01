import React from "react";
import "../styles/service1_1.css";


/**Fichier chkbox.js, ce sont les chkbox créees pour le service 1 */


//Les différentes couleurs de chaque checkbox
var couleurs = {
  0: ["#ffffcc", "#ffff99", "#ffff66", "#ffff32", "#ffff00"],
  1: ["#ffe5e5", "#ffb2b2", "#ff7f7f", "#ff6666", "#ff0000"],
  2: ["#e5f2e5", "#b2d8b2", "#7fbf7f", "#4ca64c", "#393"],
  3: ["#ffb2ff", "#ff99ff", "#ff7fff", "#ff4cff", "#ff00ff"],
  4: ["#e5cccc", "#D6ADAD", "#cc9999", "#b26666", "#a34747"],
  5: ["#e4e4e4", "#c9c9c9", "#afafaf", "#949494", "#7a7a7a"],
  6: ["#ffedcc", "#ffdb99", "#ffc966", "#ffb732", "#ffa500"],
  7: ["#ccffcc", "#99ff99", "#66ff66", "#32ff32", "#00ff00"],
  8: ["#ebdeeb", "#d7bed7", "#c39dc3", "#af7daf", "#9b5d9b"],
  9: ["#ccffff", "#99ffff", "#66ffff", "#32ffff", "#00ffff"],
  10: ["#fff2f4", "#ffe5ea", "#ffd9df", "#ffccd5", "#ffc0cb"],
  11: ["#e6e6f6", "#cdceed", "#b5b6e5", "#9c9edc", "#8486d4"],
  12: ["#F3EEEA", "#e6ded5", "#d4c5b5", "#c7b4a0", "#c1ac95"],
  13: ["#ece6f9", "#ded3f4", "#dacdf3", "#d5c7f1", "#d1c1f0"],
};

class Chkbox extends React.Component {
  constructor(props) {
    //On récupère les propriétés définis chez le parent
    super(props);
    this.name = this.props.name;
    this.identifiant = this.props.identifiant;
    this.color = this.props.color;
    this.checked = true;
    this.label = this.props.label;
  }
  //Fonction qui retourne le résultat
  render() {
    return (
      <tr>
        <td
          style={{
            display: "inline-flex",
          }}
        >
          <input
            name={this.identifiant}
            type="checkbox"
            onClick={() =>
              this.changerCouleur(this.identifiant,this.label)
            }
            defaultChecked={this.checked}
          />
          <div style={this.calcMarge(this.name)} className="chart">
            <div
              style={{
                backgroundColor: couleurs[this.label - 1][0],
                width: "20%",
              }}
            ></div>
            <div
              style={{
                backgroundColor: couleurs[this.label - 1][1],
                width: "20%",
              }}
            ></div>
            <div
              style={{
                backgroundColor: couleurs[this.label - 1][2],
                width: "20%",
              }}
            ></div>
            <div
              style={{
                backgroundColor: couleurs[this.label - 1][3],
                width: "20%",
              }}
            ></div>
            <div
              style={{
                backgroundColor: couleurs[this.label - 1][4],
                width: "20%",
                marginLeft: "-2px",
              }}
            ></div>
          </div>
          <div style={{ fontSize: "15px" }}>{this.props.name}</div>
        </td>
      </tr>
    );
  }
  //Permet d'enlever l'overflow dû au manque d'espace des labels (On gagne plus d'espaces)
  calcMarge(name) {
    if (name.includes("\n")) {
      const style = {
        marginTop: "14px",
        width: "5rem",
      };
      return style;
    }
    const style = {
      marginTop: "3px",
    };
    return style;
  }
  //Permet de changer les couleurs après le changement d'état d'une checkbox
  changerCouleur(name,labelc) {
    let propagandas = [
      "Appeal_to_Authority",
      "Appeal_to_fear-prejudice",
      "Bandwagon,Reductio_ad_hitlerum",
      "Black-and-White_Fallacy",
      "Causal_Oversimplification",
      "Doubt",
      "Exaggeration,Minimisation",
      "Flag-Waving",
      "Loaded_Language",
      "Name_Calling,Labeling",
      "Repetition",
      "Slogans",
      "Thought-terminating_Cliches",
      "Whataboutism,Straw_Men,Red_Herring",
    ];
    let colors = {
      1: ["#ffffcc", "#ffff99", "#ffff66", "#ffff32", "#ffff00"],
      2: ["#ffe5e5", "#ffb2b2", "#ff7f7f", "#ff6666", "#ff0000"],
      3: ["#e5f2e5", "#b2d8b2", "#7fbf7f", "#4ca64c", "#393"],
      4: ["#ffb2ff", "#ff99ff", "#ff7fff", "#ff4cff", "#ff00ff"],
      5: ["#e5cccc", "#D6ADAD", "#cc9999", "#b26666", "#a34747"],
      6: ["#e4e4e4", "#c9c9c9", "#afafaf", "#949494", "#7a7a7a"],
      7: ["#ffedcc", "#ffdb99", "#ffc966", "#ffb732", "#ffa500"],
      8: ["#ccffcc", "#99ff99", "#66ff66", "#32ff32", "#00ff00"],
      9: ["#ebdeeb", "#d7bed7", "#c39dc3", "#af7daf", "#9b5d9b"],
      10: ["#ccffff", "#99ffff", "#66ffff", "#32ffff", "#00ffff"],
      11: ["#fff2f4", "#ffe5ea", "#ffd9df", "#ffccd5", "#ffc0cb"],
      12: ["#e6e6f6", "#cdceed", "#b5b6e5", "#9c9edc", "#8486d4"],
      13: ["#F3EEEA", "#e6ded5", "#d4c5b5", "#c7b4a0", "#c1ac95"],
      14: ["#ece6f9", "#ded3f4", "#dacdf3", "#d5c7f1", "#d1c1f0"],
    };
    
    //On regarde si la checkbox est coché ou non
    if (document.querySelector('input[name="' + name + '"]').checked === true) {
      //On selectionne dans le résultat tous les bouts de textes ayant le même id que la checkbox
      let span = document.querySelectorAll("#res #" + name);
      span.forEach(function (sp) {
        //La couleur dépend de la probabilité, plus elle est basse plus la couleur est claire
        sp.style.backgroundColor =
          colors[labelc][
            Math.floor((sp.getAttribute("probability") * 100) / 25)
          ];
        sp.title = sp.getAttribute("title2");
      });
      //On selectionne tous les bouts de texte ayant plusieurs "couleurs"
      span = document.querySelectorAll("#res #propaganda99");
      span.forEach(function (sp) {
        //Labels est l'attribut qui nous donne les différents label appliqués à l'instant t sur la checkbox
        let labels = sp.getAttribute("labels").split(",");
        //labels2 est l'attribut qui nous donne les différents label appliqués dès l'initialisation et sont donc inchangés
        let labels2 = sp.getAttribute("labels2").split(",");
        //Si a l'initialisation la checkbox possédait cette couleur et que la checkbox a été coché alors on rajoute le label
        if (labels2.includes(labelc)) {
          labels.push(labelc);
        }
        //On enlève les vides de la chaine caractère label pour que si dans le futur on fait un split nous n'avons pas de partie vide
        if (labels.includes("")) {
          let indice = labels.indexOf("");
          labels.splice(indice, 1);
        }

        if (labels2.includes(labelc)) {
          sp.setAttribute("labels", labels);
          //Si le texte ne possède qu'un label alors il n'est pas en gras et possède une couleur de fond
          if (labels.length === 1) {
            sp.style.textDecoration = "none";
            sp.style.fontWeight = "normal";

            sp.style.backgroundColor =
              colors[labelc][
                Math.floor((sp.getAttribute("probability") * 100) / 25)
              ];
            let titre = sp.getAttribute("title2").split("\n");
            //On rajoute le label au titre
            for (let t = 0; t < titre.length; t++) {
              if (titre[t].split(" - ")[0] === propagandas[labelc - 1]) {
                sp.setAttribute(
                  "title",
                  propagandas[labelc - 1] +
                    " - " +
                    titre[t].split(" - ")[1] +
                    "\n"
                );
                sp.style.backgroundColor =
                  colors[labelc][
                    Math.floor(
                      (titre[t].split(" - ")[1].replace("Probability: ", "") *
                        100) /
                        25
                    )
                  ];
              }
            }

          } //Si le texte ne possède aucun label alors il n'a pas de fond et n'est pas en gras
          else if (labels.length === 0) {
            sp.style.background = "none";
            sp.style.textDecoration = "none";
            sp.style.fontWeight = "normal";

            sp.removeAttribute("title");
          } //Le texte possède plusieurs label, il est donc en gras
          else {
            sp.style.background = "none";
            sp.style.fontWeight = "bold";

            let titre2 = sp.getAttribute("title");
            let titre = sp.getAttribute("title2").split("\n");
            if (
              titre2 !== null &&
              titre2.indexOf(propagandas[labelc - 1]) === -1
            ) {
              for (let t = 0; t < titre.length; t++) {
                if (titre[t].split(" - ")[0] === propagandas[labelc - 1]) {
                  sp.setAttribute(
                    "title",
                    titre2 +
                      propagandas[labelc - 1] +
                      " - " +
                      titre[t].split(" - ")[1] +
                      "\n"
                  );
                }
              }
            }
          }
        }
       
      });
    } else {
      //Dans le cas ou la checkbox n'est pas coché
      let span = document.querySelectorAll("#res #" + name);

      span.forEach(function (sp) {
        sp.style.background = "none";
        sp.removeAttribute("title");
      });
      span = document.querySelectorAll("#res #propaganda99");
      span.forEach(function (sp) {
        let labels = sp.getAttribute("labels").split(",");

        let labels2 = sp.getAttribute("labels2").split(",");

        if (labels2.includes(labelc)) {
          let indice = labels.indexOf(labelc);
          labels.splice(indice, 1);
        }

        sp.setAttribute("labels", labels);

        if (labels.length === 1) {
          sp.style.textDecoration = "none";
          sp.style.fontWeight = "normal";
          let titre = "";
          if (sp.getAttribute("title") !== null) {
            titre = sp.getAttribute("title").split("\n");
          }
          let res = "";
          for (let t = 0; t < titre.length - 1; t++) {
            if (titre[t].split(" - ")[0] !== propagandas[labelc - 1]) {
              sp.style.backgroundColor =
                colors[labels[0]][
                  Math.floor(
                    (titre[t].split(" - ")[1].replace("Probability: ", "") *
                      100) /
                      25
                  )
                ];
              res = res + titre[t] + "\n";
            }
          }
          sp.setAttribute("title", res);
        } else if (labels.length === 0) {
          sp.style.textDecoration = "none";
          sp.style.fontWeight = "normal";

          sp.style.background = "none";
          sp.removeAttribute("title");
        } else {
          let titre = "";
          if (sp.getAttribute("title") !== null) {
            titre = sp.getAttribute("title").split("\n");
          }
          let res = "";
          for (let t = 0; t < titre.length - 1; t++) {
            if (titre[t].split(" - ")[0] !== propagandas[labelc - 1]) {
              res = res + titre[t] + "\n";
            }
          }
          sp.setAttribute("title", res);
          sp.style.background = "none";
          sp.style.fontWeight = "bold";
        }
      
      });
    }
  }
}

export default Chkbox;
