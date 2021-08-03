import React from "react";
import "../styles/service1_1.css";
import { changerCloud } from "../pages/service2_1";

/**File chkbox.js, these are the chkboxes created for service 2 */
class Chkbox extends React.Component {
  constructor(props) {
    super(props);
    this.identifiant = this.props.identifiant;
    this.color = this.props.color;
    this.checked = true;
  }
  render() {
    return (
      <tr>
        <td>
          <input
            name={this.identifiant}
            type="checkbox"
            onClick={() => changerCloud(this.identifiant, this.color)}
            defaultChecked={this.checked}
          />{" "}
          <mark id={this.identifiant}>{this.props.name}</mark>
        </td>
      </tr>
    );
  }
}

export default Chkbox;
