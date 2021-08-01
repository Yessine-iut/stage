import styled from "styled-components";

/**Fichier button.js, Composant button, ce sont tous les boutons crées dans chaque page du site. */

//Permet de définir la couleur du bouton
const theme = {
  blue: {
    default: "#3f51b5",
    hover: "#283593",
  },
};

//Création du bouton en CSS
const Button = styled.button`
  background-color: ${(props) => theme[props.theme].default};
  color: white;
  padding: 5px 15px;
  border-radius: 5px;
  outline: 0;
  text-transform: uppercase;
  margin: 10px 0px;
  cursor: pointer;
  box-shadow: 0px 2px 2px lightgray;
  transition: ease background-color 250ms;
  &:hover {
    background-color: ${(props) => theme[props.theme].hover};
  }
  &:disabled {
    cursor: default;
    opacity: 0.7;
  }
`;
//Par défaut le thème est bleu
Button.defaultProps = {
  theme: "blue",
};

export default Button;
