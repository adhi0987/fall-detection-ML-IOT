import "../../styles/navbar.styles.css"
const Navbar = () => (
  <nav className="bg-gray-800 p-4 fixed w-full z-10 top-0 shadow-lg">
    <div className="container mx-auto flex justify-between items-center">
      <div className="text-white text-2xl font-bold font-inter">
        Fall Detection Dashboard
      </div>
      <div className="hidden md:flex space-x-4">
        {/* Navigation links can be added here */}
      </div>
    </div>
  </nav>
);
export default Navbar;