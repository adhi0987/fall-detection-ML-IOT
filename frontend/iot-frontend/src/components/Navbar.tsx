// import "../../styles/navbar.styles.css";

// const Navbar = () => (
//   <nav className="navbar">
//     <div className="container mx-auto flex justify-between items-center">
//       <div className="navbar-brand">
//         Fall Detection Dashboard
//       </div>
//       <div className="navbar-links">
//       </div>
//     </div>
//   </nav>
// );

// export default Navbar;
// src/components/Navbar.tsx
import { NavLink } from "react-router-dom";
import "../App.css";

const Navbar = () => (
  <nav className="navbar">
    {/* This is the line that needs to be changed */}
    <div className="navbar-container"> 
      <div className="navbar-brand">
        <NavLink to="/">Fall Detection Dashboard</NavLink>
      </div>
      <div className="navbar-links">
        <NavLink to="/">Analytics</NavLink>
        <NavLink to="/training">Training & Labeling</NavLink>
      </div>
    </div>
  </nav>
);

export default Navbar;