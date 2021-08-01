import PropTypes from "prop-types";
import cloud from "d3-cloud";
import React, {
  useEffect,
  useImperativeHandle,
  forwardRef,
  useRef
} from "react";
import { select } from "d3-selection";
import { v4 as uuidv4 } from "uuid";

import { defaultFontSizeMapper } from "./defaultMappers";

const WordCloud = forwardRef((props, ref) => {
  WordCloud.propTypes = {
    data: PropTypes.arrayOf(
      PropTypes.shape({
        text: PropTypes.string.isRequired,
        value: PropTypes.number.isRequired
      })
    ).isRequired,
    font: PropTypes.oneOfType([PropTypes.string, PropTypes.func]),
    fontSizeMapper: PropTypes.func,
    height: PropTypes.number,
    padding: PropTypes.oneOfType([PropTypes.number, PropTypes.func]),
    rotate: PropTypes.oneOfType([PropTypes.number, PropTypes.func]),
    width: PropTypes.number,
    onWordClick: PropTypes.func,
    onWordMouseOut: PropTypes.func,
    onWordMouseOver: PropTypes.func,
    defaultColor: PropTypes.string,
    formatImgDownload: PropTypes.string
  };
  const svgRef = useRef();
  const className = `word-cloud-text-chart${uuidv4()}`;

  const defaultProps = {
    width: 700,
    height: 600,
    padding: 5,
    font: "serif",
    fontSizeMapper: defaultFontSizeMapper,
    rotate: 0,
    onWordClick: null,
    onWordMouseOver: null,
    onWordMouseOut: null,
    formatImgDownload: "png",
    defaultColor: "#333"
  };
  const {
    data,
    width,
    height,
    padding,
    font,
    fontSizeMapper,
    rotate,
    onWordClick,
    onWordMouseOver,
    onWordMouseOut,
    defaultColor,
    formatImgDownload
  } = props;
  const fillColor = (d, i) =>
    d.color ? d.color : defaultColor || defaultProps.defaultColor;
  const fontWeight = (d, i) => (d.fontWeight ? d.fontWeight : "normal");
  const layout = cloud()
    .size([width || defaultProps.width, height || defaultProps.height])
    .font(font || defaultProps.font)
    .words(data)
    .padding(padding || defaultProps.padding)
    .rotate(rotate || defaultProps.rotate)
    .fontSize(fontSizeMapper)
    .on("end", words => {
      const texts = select(`div.${className}`)
        .append("svg")
        .attr("width", layout.size()[0])
        .attr("height", layout.size()[1])
        .append("g")
        .attr(
          "transform",
          `translate(${layout.size()[0] / 2},${layout.size()[1] / 2})`
        )
        .selectAll("text")
        .data(words)
        .enter()
        .append("text")
        .style("font-size", d => `${d.size}px`)
        .style("font-family", font)
        .style("fill", fillColor)
        .style("font-weight", fontWeight)
        .attr("text-anchor", "middle")
        .attr("transform", d => `translate(${[d.x, d.y]})rotate(${d.rotate})`)
        .text(d => d.text);

      if (onWordClick) {
        texts.on("click", onWordClick);
      }
      if (onWordMouseOver) {
        texts.on("mouseover", onWordMouseOver);
      }
      if (onWordMouseOut) {
        texts.on("mouseout", onWordMouseOut);
      }
    });
  useEffect(() => {
    layout.start();
  }, [data]);

  // convert svg to base64 image
  const loadPngData = ({ context, dataImg, format, canvas }) =>
    new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        context.clearRect(
          0,
          0,
          width || defaultProps.width,
          height || defaultProps.height
        );
        context.drawImage(
          image,
          0,
          0,
          width || defaultProps.width,
          height || defaultProps.height
        );
        const pngData = canvas.toDataURL("image/" + format);
        return resolve(pngData);
      };
      image.src = dataImg;
    });

  useImperativeHandle(ref, () => ({
    async toBase64Image() {
      const format = formatImgDownload
        ? formatImgDownload
        : defaultProps.formatImgDownload;
      const imgString = new XMLSerializer().serializeToString(
        svgRef.current.querySelector("svg")
      );
      const dataImg = `data:image/svg+xml;base64,${window.btoa(imgString)}`;
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      canvas.width = width || defaultProps.width;
      canvas.height = height || defaultProps.height;
      const data = await loadPngData({ dataImg, context, format, canvas });
      return data;
    }
  }));
  // render based on new data
  return <div className={className} ref={svgRef}></div>;
});

export default WordCloud;
