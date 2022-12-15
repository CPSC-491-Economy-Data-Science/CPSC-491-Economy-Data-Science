import React, { Component }  from 'react';
import Navbar from "./Navbar";
import Home from "./pages/Home";
import About from "./pages/About";
import Models from "./pages/Models";
import Resources from "./pages/Resources";

function App() {
  let component 
  switch (window.location.pathname) {
    case "/":
      component = <Home />;
      break;
    case "/Models":
      component = <Models />;
      break;  
    case "/Resources":
      component = <Resources />;
      break;
    case "/About":
      component = <About />;
      break;
  }
  return (
    <>
      <Navbar />
      {component}
    </>
  )
}

export default App;