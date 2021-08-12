import React, { useState } from "react";
import Button from "./button";

import "../styles/textInput.css";
import axios from "axios";
import { Link, useHistory } from "react-router-dom";

/**File textInput.js, this is the textInput used in service 1 with the buttons as well as the examples */

export var data;
let id2 = 0;

function App() {
  let history = useHistory();
  const [input, setInput] = useState("");
  const ColoredLine = ({ color }) => (
    <hr
      style={{
        color: color,
        backgroundColor: color,
        height: 2,
        width: 900,
      }}
    />
  );

  function sendData() {
    // We make all the elements disappear and the loading GIF appears
    let img = document.getElementById("loading");
    img.style.display = "flex";

    let div = document.getElementById("divF");
    div.style.display = "none";
    div = document.getElementById("h1");
    div.style.display = "none";
    div = document.getElementById("footer");
    div.style.display = "none";
    let h1 = document.getElementsByTagName("h1");
    for (let i = 0; i < h1.length; i++) {
      h1[i].style.display = "none";
    }
    let h2 = document.getElementsByTagName("h2");
    for (let i = 0; i < h1.length; i++) {
      h2[i].style.display = "none";
    }

    if (!input.trim()) {
      data = " ";
      let div = document.getElementById("footer");

      div.style = " display: visible";
      history.push("/PropagandaTechniquesClassification2");
    } else {
      id2++;
      axios
        .post("http://127.0.0.2:5000/post", {
          id: id2,
          text: input,
          headers: { "Access-Control-Allow-Origin": "*" },
        })
        .then((response) => {
          data = response.data;
          let div = document.getElementById("footer");

          div.style = " display: visible";
          history.push("/PropagandaTechniquesClassification2");
        })

        .catch((error) => {
          console.log(error);
        });
    }
  }

  return (
    <>
      <div>
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
              Propaganda techniques classification
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
              Type in some text or select an example below
            </h2>
          </div>
        </div>
      </div>
      <div
        id="divF"
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          flexDirection: "column",
        }}
      >
        <textarea
          value={input}
          className="textArea"
          onChange={(e) => {
            setInput(e.target.value);
          }}
        />
        <div>
          <Button
            id="boutonData"
            disabled={input === ""}
            onClick={() => sendData()}
            style={{ marginRight: "10px" }}
          >
            Analyze
          </Button>

          <Link to="/">
            <Button
              style={{
                width: "98.48px",
              }}
            >
              back
            </Button>
          </Link>
        </div>
        <ColoredLine color="white" />
        <div
          style={{
            width: "58%",
            backgroundColor: "#D9D9D9",
            marginBottom: "2%",
          }}
        >
          <div
            style={{
              textAlign: "center",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              flexDirection: "column",
              width: "150%",
              marginLeft: "-25%",
            }}
          >
            <h3 style={{ marginTop: "20px" }}>Textual examples</h3>
            <h4
              className="textExample"
              style={{ fontSize: "1rem", marginBottom: "25px" }}
            >
              Click one a sample below to copy and try to identify propaganda
              snippets.
            </h4>
            <div
              className="col-container"
            >
              <div className="col">
                <div
                  id="box"
                  content="Manchin says Democrats acted like babies at the SOTU (video) Personal Liberty Poll Exercise your right to vote."
                  onClick={(element) => updateContent(element)}
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "2px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  Manchin says Democrats acted like babies at the SOTU (video)
                  Personal Liberty Poll Exercise your right to vote.
                </div>
                <div
                  id="box"
                  onClick={(element) => updateContent(element)}
                  content="Democrat West Virginia Sen. Joe Manchin says his colleagues’ refusal to stand or applaud during President Donald Trump’s State of the Union speech was disrespectful and a signal that the party is more concerned with obstruction than it is with progress."
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "2px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  Democrat West Virginia Sen. Joe Manchin says his colleagues’
                  refusal to stand or applaud during President Donald Trump’s
                  State of the Union speech was disrespectful and a signal that
                  the party is more concerned with obstruction than it is with
                  progress.
                </div>
                <div
                  id="box"
                  onClick={(element) => updateContent(element)}
                  content="Another Republican who wants to stop investigating is former RNC Chair, Michael Steele: “There is no Spygate because there are no spies in the campaign.” The attitude of Steve Scalise is wishy-washy."
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "2px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  Another Republican who wants to stop investigating is former
                  RNC Chair, Michael Steele: “There is no Spygate because there
                  are no spies in the campaign.” The attitude of Steve Scalise
                  is wishy-washy.
                </div>
              </div>
              <div className="col">
                <div
                  id="box"
                  onClick={(element) => updateContent(element)}
                  content="“It’s not necessary to agree with everything he says, it’s not necessary to approve of everything he does, but history will judge him as being on the right side of a struggle between good and evil,” Batten continued."
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "2px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  “It’s not necessary to agree with everything he says, it’s not
                  necessary to approve of everything he does, but history will
                  judge him as being on the right side of a struggle between
                  good and evil,” Batten continued.
                </div>
                <div
                  id="box"
                  onClick={(element) => updateContent(element)}
                  content="She noticed that Latino voters did record-breaking numbers, especially in states like Florida, Nevada and Arizona – the last of which she describes as “a key state for us.” She brags that the company used its power to ensure that millions of people saw certain hashtags and social media impressions, with the goal of influencing their behavior during the election.”"
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "2px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  She noticed that Latino voters did record-breaking numbers,
                  especially in states like Florida, Nevada and Arizona – the
                  last of which she describes as “a key state for us.” She brags
                  that the company used its power to ensure that millions of
                  people saw certain hashtags and social media impressions, with
                  the goal of influencing their behavior during the election.”
                </div>
                <div
                  id="box"
                  onClick={(element) => updateContent(element)}
                  content="Meanwhile, Google has yet to answer why their search results for the word “Idiot” are vastly different from DuckDuckGo."
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "2px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  Meanwhile, Google has yet to answer why their search results
                  for the word “Idiot” are vastly different from DuckDuckGo.
                </div>
              </div>
              <div className="col">
                <div
                  id="box"
                  onClick={(element) => updateContent(element)}
                  content="But Democrats,including the committee’s ranking member, Rep. Adam Schiff, D-Calif., are blasting the declassification order as a “clear abuse of power.”"
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "2px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  But Democrats,including the committee’s ranking member, Rep.
                  Adam Schiff, D-Calif., are blasting the declassification order
                  as a “clear abuse of power.”
                </div>
                <div
                  id="box"
                  onClick={(element) => updateContent(element)}
                  content="I'm all for situations like Nikolas Cruz who assaulted teachers and students being arrested and tried and dealt with lawfully, but to go seizing people's property and denying them their liberty base on the hearsay of another person is anti-constitutional and anti-American."
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "2px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  I'm all for situations like Nikolas Cruz who assaulted
                  teachers and students being arrested and tried and dealt with
                  lawfully, but to go seizing people's property and denying them
                  their liberty base on the hearsay of another person is
                  anti-constitutional and anti-American.
                </div>

                <div
                  id="box"
                  onClick={(element) => updateContent(element)}
                  content="In a glaring sign of just how stupid and petty things have become in Washington these days, Manchin was invited on Fox News Tuesday morning to discuss how he was one of the only Democrats in the chamber for the State of the Union speech not looking as though Trump killed his grandma."
                  style={{
                    backgroundColor: "#F8F8F8",
                    marginBottom: "20px",
                    border: "solid 1px",
                    padding: "15px",
                    fontFamily: "Arial Unicode MS, Arial, sans-serif",
                    fontSize: "13px",
                  }}
                >
                  In a glaring sign of just how stupid and petty things have
                  become in Washington these days, Manchin was invited on Fox
                  News Tuesday morning to discuss how he was one of the only
                  Democrats in the chamber for the State of the Union speech not
                  looking as though Trump killed his grandma.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
  function updateContent(params) {
    setInput(params.target.getAttribute("content"));
    window.scrollTo({
      top: 0,
      left: 0,
      behavior: "smooth",
    });
  }
}

export default App;
