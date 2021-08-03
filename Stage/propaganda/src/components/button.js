import styled from "styled-components";

/**File button.js, Composant button, these are all the buttons created in each page of the website. */

// Used to define the color of the button
const theme = {
  blue: {
    default: "#3f51b5",
    hover: "#283593",
  },
};

// Create the button in CSS
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
// By default the theme is blue
Button.defaultProps = {
  theme: "blue",
};

export default Button;
