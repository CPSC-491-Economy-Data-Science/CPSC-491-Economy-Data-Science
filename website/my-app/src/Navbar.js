export default function Navbar(){
    return (
    <nav className="nav">
        <a href="/" className="site-title">
            Econ Project
        </a>
        <ul>
            <CustomLink href="/Models">Models</CustomLink>
            <CustomLink href="/Resources">Resources</CustomLink>
            <CustomLink href="/About">About</CustomLink>
        </ul>
    </nav>
    )
}

function CustomLink({ href, children, ...props}) {
    const path = window.location.pathname;
    return (
        <li className={path === href ? "active" : ""}>
            <a href={href} {...props}>{children}</a>
        </li>
    )
}