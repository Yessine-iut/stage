# react-d3-cloud

[![NPM version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Dependency Status][david_img]][david_site]

A word cloud react component built with [d3-cloud](https://github.com/jasondavies/d3-cloud).

![image](https://cloud.githubusercontent.com/assets/6868283/20619528/fa83334c-b32f-11e6-81dd-6fe4fa6c52d9.png)

## Usage

```sh
npm install @nhuson/react-d3-cloud
```

```jsx
import React from "react";
import { render } from "react-dom";
import WordCloud from "@nhuson/react-d3-cloud";

const data = [
  { text: "Hello", value: 1000, color: "grey", fontWeight: 500 },
  { text: "lol", value: 200, color: "grey", fontWeight: "normal" },
  { text: "first impression", value: 800, color: "#ccc", fontWeight: "bold" },
  { text: "very cool", value: 1000000 },
  { text: "duck", value: 10 }
];

const fontSizeMapper = word => Math.log2(word.value) * 5;
const rotate = word => word.value % 360;

render(
  <WordCloud data={data} fontSizeMapper={fontSizeMapper} rotate={rotate} />,
  document.getElementById("root")
);
```

for more detailed props, please refer to below:

## Props

| name            | description                                                                                                  | type                                            | required | default               |
| --------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- | -------- | --------------------- |
| data            | The input data for rendering                                                                                 | Array<{ text: string, value: number }>          | ✓        |
| width           | Width of component (px)                                                                                      | number                                          |          | 700                   |
| height          | Height of component (px)                                                                                     | number                                          |          | 600                   |
| fontSizeMapper  | Map each element of `data` to font size (px)                                                                 | Function: `(word: string, idx: number): number` |          | `word => word.value;` |
| rotate          | Map each element of `data` to font rotation degree. Or simply provide a number for global rotation. (degree) | Function \| number                              |          | 0                     |
| padding         | Map each element of `data` to font padding. Or simply provide a number for global padding. (px)              | Function \| number                              |          | 5                     |
| font            | The font of text shown                                                                                       | Function \| string                              |          | serif                 |
| onWordClick     | Function called when click event triggered on a word                                                         | Function: `word => {}`                          |          | null                  |
| onWordMouseOver | Function called when mouseover event triggered on a word                                                     | Function: `word => {}`                          |          | null                  |
| onWordMouseOut  | Function called when mouseout event triggered on a word                                                      | Function: `word => {}`                          |          | null                  |
| defaultColor  | Default color text, if not set color in data                                                    | String                   |          | #333                  |
| formatImgDownload  | Format image base64                                                      | String                        |          | png                  |
## Convert chart to base64 image

```jsx
import React, { useRef, useEffect } from "react";
import { render } from "react-dom";
import WordCloud from "@nhuson/react-d3-cloud";

const data = [
  { text: "Hello", value: 1000, color: "grey", fontWeight: 500 },
  { text: "lol", value: 200, color: "grey", fontWeight: "normal" },
  { text: "first impression", value: 800, color: "#ccc", fontWeight: "bold" },
  { text: "very cool", value: 1000000 },
  { text: "duck", value: 10 }
];
const Cloud = (props) => {
    const cloudRef = useRef();
    useEffect(() => {
        cloudRef.current.toBase64Image().then(imgBase64 => console.log(imgBase64))
    }, [])
    return (
         <WordCloud
          height={300}
          data={data}
          padding={10}
          ref={cloudRef}
    />
    )
}

```

## Build

```sh
npm run build
```

## Test

### pre-install

#### Mac OS X

```sh
brew install pkg-config cairo pango libpng jpeg giflib librsvg
npm install
```