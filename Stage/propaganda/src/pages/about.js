
import "../styles/about.css";
// The About page of the website
const About = () => {
  return (
    <>
      <div style={{}}>
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
              About
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
              "Disinformation is spreading online through news, social media and
              etc. Based on NLP techniques, we explore such disinformation via
              rhetorical, psychological and strategies to detect such propaganda
              text."
            </h2>
          </div>
        </div>

        <p style={{ marginLeft: "2%", marginRight: "2%" }}>
          One of the mechanisms through which disinformation is spreading
          online, in particular through social media, is by employing propaganda
          techniques. These include specific rhetorical and psychological
          strategies, ranging from leveraging on emotions to exploiting logical
          fallacies. We adopt a supervised approach (i.e., BERT and RoBERTa
          Transformer-based models) to classify textual snippets both as
          propaganda messages and according to the precise applied propaganda
          technique, as well as a detailed linguistic analysis of the features
          characterising propaganda information in text (e.g., semantic,
          sentiment and argumentation features). We provide two automatized
          services concerning propaganda snippets in our detection system.
        </p>
        <h2
          style={{
            fontSize: "20px",
            marginLeft: "2%",
            textDecoration: "underline",
            backgroundColor: "white",
            textAlign: "center",
            marginTop: "50px",
          }}
        >
          Propaganda techniques classification
        </h2>
        <p style={{ marginLeft: "2%", marginRight: "2%" }}>
          We let users freely input a text to employ our supervised system to
          display precisely propaganda snippets if it exists in the text, along
          with (multiple) probability of propaganda technique used. Users can
          also download their outputs as a JSON file for academic usages.
        </p>
        <h2
          style={{
            fontSize: "20px",
            marginLeft: "2%",
            textDecoration: "underline",
            backgroundColor: "white",
            textAlign: "center",
            marginTop: "50px",
          }}
        >
          Propaganda Word Clouds
        </h2>
        <p style={{ marginLeft: "2%", marginRight: "2%" }}>
          {" "}
          We employ our supervised system to display word clouds from a free
          text. Users can observe easily the most important snippets regarding
          its propaganda techniques and probabilities.{" "}
        </p>
        <p style={{ marginLeft: "2%", marginRight: "2%" }}>
          Our system consists of two processes which employ a supervised
          approach to proceed the prediction at sentence-level. We firstly
          detect the propaganda snippets of the given input as our first step,
          where in this process we obtain 0.88 macro F1-score. As the last part
          of our system, we employ a supervised classifier to perform a
          classification on 14 propaganda techniques [1]. The result of this
          part of our system obtains in average of all techniques is of ~0.90
          macro F1-score. Our system was trained and tested based on
          academic-used resources from two computational linguistic workshops
          namely,
          <a
            className="linkColor"
            href="https://propaganda.qcri.org/nlp4if-shared-task/"
          >
            {" "}
            NLP4IF’2019
          </a>
          , and{" "}
          <a
            className="linkColor"
            href="https://propaganda.qcri.org/ptc/index.html"
          >
            SemEval2020 Task-11
          </a>
          .
        </p>
        <p style={{ marginLeft: "2%", marginRight: "2%" }}>
          [1] List of propaganda techniques in this work.
        </p>
        <ol style={{ marginLeft: "2%", marginRight: "2%" }}>
          <li>Appeal_to_Authority</li>
          <li>Appeal_to_fear-prejudice</li>
          <li>Bandwagon,Reductio_ad_hitlerum</li>
          <li>Black-and-White_Fallacy</li>
          <li>Causal_Oversimplification</li>
          <li>Doubt</li>
          <li>Exaggeration,Minimisation</li>
          <li>Flag-Waving</li>
          <li>Loaded_Language</li>
          <li>Name_Calling,Labeling</li>
          <li>Repetition</li>
          <li>Slogans</li>
          <li>Thought-terminating_Cliches</li>
          <li>Whataboutism,Straw_Men,Red_Herring</li>
        </ol>
      </div>
    </>
  );
};

export default About;
