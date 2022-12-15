import background from '../images/frontpage-background.png'

export default function Home() {
    return (
        <div>
            {/* <h1>hi</h1>
            <h1 className="learn-btn">Explore Data Models</h1>
            <a href="#Learn"><button>LEARN MORE</button></a> */}
            <img className="frontpage-background" src={background}/>
            {/*<h1 className="econ-font">Econ Project</h1> */}
        </div>
    )
}