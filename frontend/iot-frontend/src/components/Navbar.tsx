import "../../styles/navbar.styles.css";

const Navbar = () => (
  <nav className="navbar">
    <div className="container mx-auto flex justify-between items-center">
      <div className="navbar-brand">
        Fall Detection Dashboard
      </div>
      <div className="navbar-links">
        {/* Navigation links can be added here */}
      </div>
    </div>
  </nav>
);

export default Navbar;