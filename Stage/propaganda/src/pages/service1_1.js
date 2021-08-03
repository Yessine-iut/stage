import React from "react";
import { data } from "../components/textInput";
import "../styles/service1_1.css";
import Button from "../components/button";
import Multiplechkbox from "../components/multipleChkbox";
import { Link } from "react-router-dom";
import { saveAs } from "file-saver";
import notFound from "../images/404.png";

/**Service1_1.js file, this is the site's service 1 */
class Service1_1 extends React.Component {
  //The render of the page
  render() {
    if (data === " ") {
      return (
        <>
          <div
            style={{ textAlign: "center", marginTop: "10%", fontSize: "60px" }}
          >
            The input text is empty.
          </div>
          <div style={{ textAlign: "center", fontSize: "20px" }}>
            Please go back and put some text.
          </div>
          <Link to="/PropagandaSnippetsDetection">
            <Button style={{ marginLeft: "48%", marginBottom: "80px" }}>
              back
            </Button>
          </Link>
        </>
      );
    }
    if (data === undefined) {
      return (
        <>
          {" "}
          <img
            style={{
              display: "block",
              margin: "auto",
              textAlign: "center",
              alignItems: "center",
              marginTop: "10%",
            }}
            src={notFound}
            alt="Error 404"
          />
          <Link to="/">
            <Button style={{ marginLeft: "48%", marginBottom: "80px" }}>
              back
            </Button>
          </Link>
        </>
      );
    }
    return (
      <div style={{ height: "700px" }}>
        <div>
          <div
            id="res"
            style={{
              lineHeight: "25px",
              width: "76%",
            }}
          >
            <div
              style={{
                marginTop: "70px",
                backgroundColor: "white",
                minHeight: "500px",
                position: "relative",
                maxHeight: "500px",
                overflowY: "scroll",
                paddingLeft: "8px",
              }}
            >
              {texte()}
            </div>
            <div
              id="btn"
              className="button"
              style={{
                position: "relative",
                left: "42%",
                top: "2em",
                padding: "4px 5px",
              }}
            >
              <Button
                style={{ marginRight: "10px" }}
                onClick={() => saveDynamicDataToFile()}
              >
                Export JSON file
              </Button>

              <Link to="/">
                <Button>back</Button>
              </Link>
            </div>
          </div>
          <div
            id="test"
            style={{
              position: "absolute",
              left: "40%",
            }}
          ></div>
          <div
            style={{
              position: "absolute",
              left: "80%",
            }}
          >
            <Multiplechkbox />
          </div>
        </div>
      </div>
    );
  }
}
//the function that creates the text by highlighting the propaganda
function texte() {
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
  let spans = [];
  let res = [];
  let tailleSpan = 0;
  let text = "";
  let name = "";
  for (let i in data) {
    name = i;
  }
  //Browse the differents propagandas
  for (let i in data[name]) {
    tailleSpan = Object.keys(data[name][i]).length;
    text = data[name][i].text;
    //Browse the differents spans
    for (let v = 1; v < tailleSpan; v++) {
      spans.push(data[name][i]["span_" + v]);
    }
    //We sort the spans with their start_char
    spans.sort(function compare(a, b) {
      if (
        typeof a.start_char !== "object" &&
        typeof b.start_char !== "object"
      ) {
        if (a.start_char < b.start_char) return -1;
        if (a.start_char > b.start_char) return 1;
        return 0;
      } else if (
        typeof a.start_char === "object" &&
        typeof b.start_char !== "object"
      ) {
        if (a.start_char[0] < b.start_char) return -1;
        if (a.start_char[0] > b.start_char) return 1;
        return 0;
      } else if (
        typeof a.start_char !== "object" &&
        typeof b.start_char === "object"
      ) {
        if (a.start_char < b.start_char[0]) return -1;
        if (a.start_char > b.start_char[0]) return 1;
        return 0;
      } else if (
        typeof a.start_char === "object" &&
        typeof b.start_char === "object"
      ) {
        if (a.start_char[0] < b.start_char[0]) return -1;
        if (a.start_char[0] > b.start_char[0]) return 1;
        return 0;
      } else return 0;
    });
    let curseur = 0;
    //Browse the spans
    for (let v = 0; v < spans.length; v++) {
      let spanCourant = spans[v];
      let spanText = spans[v].span;
      //The case where don't have a multiple propaganda spans and the length it's 2
      if (tailleSpan - 1 === 1 && typeof spanCourant.label !== "object") {
        res.push(text.substring(0, spanCourant.start_char));
        let id = "propaganda" + spans[v].label;
        let label1 = [];
        label1.push(spans[v].label);
        let couleuri = intensite(spans[v].label, spans[v].probability);
        let title =
          propagandas[spans[v].label - 1] +
          " - Probability: " +
          spans[v].probability;
        //Remove the title when the span dosn't have color
        if (couleuri === "none") {
          res.push(
            <mark
              style={{
                backgroundColor: couleuri,
              }}
              probability={spans[v].probability}
              key={v + id + label1 + title + spanCourant.span + curseur}
              id={id}
              labels={label1}
            >
              {spanCourant.span}{" "}
            </mark>
          );
        } else
          res.push(
            <mark
              style={{ backgroundColor: couleuri }}
              key={v + id + label1 + title + spanText + curseur}
              id={id}
              couleur={couleuri}
              labels={label1}
              title={title}
              title2={title}
              probability={spans[v].probability}
            >
              {spanText}
            </mark>
          );
        res.push(text.substring(spanCourant.end_index, text.length) + " ");
      } else {
        // We see if it overlaps
        let over = 0;

        if (typeof spanCourant.label === "object") {
          let SpansOverlap = [];
          let p = v;
          while (p < tailleSpan - 1) {
            if (spans[p].probability !== 0) {
              SpansOverlap.push(spans[p]);
            }

            p++;
          }
          intervalles(SpansOverlap, text, res, curseur);

          over = 1;
          v = tailleSpan;
        }
        //If it's not overlap
        if (over !== 1) {
          over = 0;
          let id = "propaganda" + spanCourant.label;
          let label1 = [];
          label1.push(spanCourant.label);
          let title =
            propagandas[spans[v].label - 1] +
            " - Probability: " +
            spans[v].probability;

          res.push(text.substring(curseur, spanCourant.start_char));
          let couleuri = intensite(spans[v].label, spans[v].probability);
          if (couleuri === "none") {
            res.push(
              <mark
                style={{
                  backgroundColor: couleuri,
                }}
                probability={spans[v].probability}
                key={v + id + label1 + title + spanCourant.span + curseur}
                id={id}
                labels={label1}
              >
                {spanCourant.span}{" "}
              </mark>
            );
          } else
            res.push(
              <mark
                style={{
                  backgroundColor: couleuri,
                }}
                probability={spans[v].probability}
                key={v + id + label1 + title + spanCourant.span + curseur}
                id={id}
                labels={label1}
                title={title}
                title2={title}
              >
                {spanCourant.span}{" "}
              </mark>
            );
          curseur = spanCourant.end_index;
          if (
            spanCourant.end_index < text.length - 1 &&
            v + 1 > spans.length - 1
          ) {
            res.push(text.substring(spanCourant.end_index, text.length));
          }
        }
      }
    }

    curseur = 0;
    tailleSpan = 0;
    spans.splice(0, spans.length);
  }
  return res;
}
//Allow to have the color thanks to label and probability
function intensite(label, probability) {
  if (label === 0) {
    return "none";
  }
  var obj1 = {
    1: ["#ffffcc", "#ffff99", "#ffff66", "#ffff32", "#ffff00"],
    2: ["#ffe5e5", "#ffb2b2", "#ff7f7f", "#ff6666", "#ff0000"],
    3: ["#e5f2e5", "#b2d8b2", "#7fbf7f", "#4ca64c", "#393"],
    4: ["#ffb2ff", "#ff99ff", "#ff7fff", "#ff4cff", "#ff00ff"],
    5: ["#e5cccc", "#D6ADAD", "#cc9999", "#b26666", "#ad3333"],
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

  probability = Math.round((probability * 100) / 25);
  return obj1[label][probability];
}
//Allow to download the JSON file
function saveDynamicDataToFile() {
  var blob = new Blob([JSON.stringify(data)], {
    type: "json/plain;charset=utf-8",
  });
  let today = new Date();
  let name =
    "PropagandaSnippetsDetection_" +
    today.getHours() +
    "h" +
    today.getMinutes() +
    "_" +
    today.getMonth() +
    "_" +
    today.getDate() +
    "_" +
    today.getFullYear();
  saveAs(blob, name);
}
//Return the text with propaganda highligh for the propaganda overlap
function intervalles(spans, texte, res, curseur) {
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

  let inter = [];
  //we will get all the start and end of the indexes
  for (let j = 0; j < spans.length; j++) {
    if (typeof spans[j].label === "object") {
      for (let i = 0; i < spans[j].label.length; i++) {
        inter.push(spans[j].start_char[i]);
        inter.push(spans[j].end_index[i]);
      }
    } else {
      inter.push(spans[j].start_char);
      inter.push(spans[j].end_index);
    }
  }

  const byValue = (a, b) => a - b;
  //We sort the indexes
  const sorted = [...inter].sort(byValue);
  //We push the first part of the sentance
  res.push(texte.substring(curseur, sorted[0]));
  curseur = sorted[0];
  let labels1 = [];

  for (let i = 1; i < sorted.length; i++) {
    //We get the labesls
    labels1 = labels(spans, curseur + 1);
    //If it's not a multiple propaganda
    if (labels1.length === 1) {
      let id = "propaganda" + labels1[0];
      if (curseur !== sorted[i]) {
        let label = labels(spans, curseur + 1);
        let probability = probabilities(spans, curseur + 1);
        let title = propagandas[label - 1] + " - Probability: " + probability;
        let couleuri = intensite(label, probability);

        res.push(
          <mark
            probability={probability}
            style={{
              backgroundColor: couleuri,
            }}
            key={
              id + labels(spans, curseur + 1) + probability + title + curseur
            }
            title={title}
            title2={title}
            id={id}
            labels={label}
            labels2={label}
          >
            {texte.substring(curseur, sorted[i])}
          </mark>
        );
      }
    } //If the part of the sentance dosn't have a propaganda
    else if (labels1.length === 0) {
      res.push(texte.substring(curseur, sorted[i]));
    } //If we have multiple propaganda
    else {
      let id = "propaganda" + 99;
      if (curseur !== sorted[i]) {
        let label = labels(spans, curseur + 1);
        let probability = probabilities(spans, curseur + 1);
        let title = "";
        for (let f = 0; f < label.length; f++) {
          if (f !== label.length - 1) {
            title =
              title +
              propagandas[label[f] - 1] +
              " - Probability: " +
              probability[f] +
              "\n";
          } else
            title =
              title +
              propagandas[label[f] - 1] +
              " - Probability: " +
              probability[f] +
              "\n";
        }
        res.push(
          <mark
            probability={probability}
            key={
              id + labels(spans, curseur + 1) + probabilities + title + curseur
            }
            id={id}
            labels={label}
            labels2={label}
            title={title}
            title2={title}
          >
            {texte.substring(curseur, sorted[i])}
          </mark>
        );
      }
    }
    //We update the cursor
    curseur = sorted[i];
    //We remove the span that we treated
    labels1.splice(0, labels1.length);

  }
  //Last part
  labels1 = labels(spans, curseur + 1);
  //If the last part of the sentance hasn't span
  if (labels1.length === 0) {
    res.push(texte.substring(curseur, texte.length));
  }   //If the last part of the sentance has one span
  else if (labels1.length === 1) {
    let id = "propaganda" + labels1[0];
    let label = labels(spans, curseur + 1);
    let probability = probabilities(spans, curseur + 1);
    let title = "";

    title =
      title + propagandas[label - 1] + " - Probability: " + probability + "\n";
    let couleuri = intensite(label, probability);
    if (couleuri === "none") {
      res.push(
        <mark
          style={{
            backgroundColor: couleuri,
          }}
          probability={probability}
          key={id + labels(spans, curseur + 1)}
          id={id}
          labels={labels(spans, curseur + 1)}
          labels2={labels(spans, curseur + 1)}
        >
          {texte.substring(curseur, texte.length)}
        </mark>
      );
    } else
      res.push(
        <mark
          style={{
            backgroundColor: couleuri,
          }}
          probability={probability}
          key={id + labels(spans, curseur + 1)}
          id={id}
          labels={labels(spans, curseur + 1)}
          labels2={labels(spans, curseur + 1)}
          title={title}
          title2={title}
        >
          {texte.substring(curseur, texte.length)}
        </mark>
      );
  } //Multiple propaganda
  else {
    let id = "propaganda" + 99;
    let label = labels(spans, curseur + 1);
    let probability = probabilities(spans, curseur + 1);
    let title = "";
    for (let f = 0; f < label.length; f++) {
      title =
        title +
        propagandas[label[f] - 1] +
        " - Probability: " +
        probability[f] +
        "\n";
    }
    res.push(
      <mark
        probability={probability}
        key={id + labels(spans, curseur + 1)}
        id={id}
        labels={labels(spans, curseur + 1)}
        labels2={labels(spans, curseur + 1)}
        title={title}
        title2={title}
      >
        {texte.substring(curseur, texte.length)}
      </mark>
    );
  }
}
//Allow to have the labels thanks to spans and the curseur in the sentance
function labels(spans, curseur) {
  let labels = [];
  for (let t = 0; t < spans.length; t++) {
    if (typeof spans[t].label === "object") {
      for (let i = 0; i < spans[t].label.length; i++) {
        if (
          curseur >= spans[t].start_char[i] &&
          curseur <= spans[t].end_index[i]
        ) {
          labels.push(spans[t].label[i]);
        }
      }
    } else {
      if (curseur >= spans[t].start_char && curseur <= spans[t].end_index) {
        labels.push(spans[t].label);
      }
    }
  }
  return labels;
}
//Allow to have the probability thanks to spans and the curseur in the sentance
function probabilities(spans, curseur) {
  let labels = [];
  for (let t = 0; t < spans.length; t++) {
    if (typeof spans[t].label === "object") {
      for (let i = 0; i < spans[t].label.length; i++) {
        if (
          curseur >= spans[t].start_char[i] &&
          curseur <= spans[t].end_index[i]
        ) {
          labels.push(spans[t].probability[i]);
        }
      }
    } else {
      if (curseur >= spans[t].start_char && curseur <= spans[t].end_index) {
        labels.push(spans[t].probability);
      }
    }
  }
  return labels;
}

export default Service1_1;
