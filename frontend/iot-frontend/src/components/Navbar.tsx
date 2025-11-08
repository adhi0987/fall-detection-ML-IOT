import { NavLink } from "react-router-dom";
import "../App.css";

const Navbar = () => (
  <nav className="navbar">
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